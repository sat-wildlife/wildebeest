

from skimage import exposure

import os
import rasterio
#import rasterio.warp             # Reproject raster samples
from rasterio import windows
#import geopandas as gps
#import PIL.Image
#import PIL.ImageDraw
import fiona


from shapely.geometry import Point
from shapely.geometry import mapping
import numpy as np               # numerical array manipulation
#from tqdm import tqdm


from skimage import measure
from sklearn.cluster import KMeans
from fiona.crs import from_epsg


from pathlib import Path

from itertools import product
from tqdm import tqdm

# !pip install ipython-autotime
# %load_ext autotime

PATCH_SIZE = 336
INPUT_BANDS = [0,1,2]

def linear_stretch(image):
    C = image.shape
    image2 = image.copy()
    for i in range(image2.shape[2]):
        p2, p98 = np.percentile(image2[:, :, i], (0.5, 99.5))
        image2[:, :, i] = exposure.rescale_intensity(image2[:, :, i],
                                                      in_range=(p2, p98))

    return image2


def get_images_to_predict(target_dir, target_filelist=None, ext=None):
    """ Get all input images to predict

    Either takes only the images specifically listed in a text file at target_filelist,
    or all images in target_dir with the correct prefix and file type
    """
    input_images = []
    if target_filelist is not None and os.path.exists(target_filelist):
        for line in open(target_filelist):
            if os.path.isabs(line.strip()) and os.path.exists(line.strip()):      # absolute paths
                input_images.append(line.strip())
            elif os.path.exists(os.path.join(target_dir, line.strip())):          # relative paths
                input_images.append(os.path.join(target_dir, line.strip()))

        print(f"Found {len(input_images)} images in {target_filelist}.")
    else:
        for file in os.listdir(target_dir):
            item_path = os.path.join(target_dir, file)
            if os.path.isfile(item_path):
                    if ext is not None:
                        if file.endswith(ext):
                            input_images.append(item_path)
                    else:
                        input_images.append(item_path)

        print(f"Found {len(input_images)} valid images in {target_dir}.")
    if len(input_images) == 0:
        raise Exception("No images to predict.")

    return sorted(input_images)

def get_tiles(ds, width=256, height=256):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

def split_image(base_dir, out_path, out_fn, tile_width=5000, tile_height=5000):
    with rasterio.open(base_dir) as inds:
        meta = inds.meta.copy()
        for window, transform in get_tiles(inds, tile_width, tile_height ):
            #print(window)
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            outpath = os.path.join(out_path,out_fn.format(int(window.col_off), int(window.row_off)))
            print(outpath)

            if Path(outpath).is_file() == True:
              print(f"Tile {outpath} already exists. Skip.")
              continue
            with rasterio.open(outpath, 'w', **meta) as outds:
                outds.write(inds.read(window=window))

def ImageToPoints(image, mask, animal_size):
    # find the contours of the image segments
    image = image.astype(np.uint8)

    transform = mask.meta['transform']
    wildebeest = []
    labels = measure.label(image)
    regions = measure.regionprops(labels)
    del labels
    for region in regions:
        #print(region.centroid)
        if region.area < 1:
            continue
        num = np.uint8(np.ceil(region.area/animal_size))
        if num == 1:
            centroid = list(np.round(region.centroid))
            wildebeest.append(centroid) #Be aware that .append() is different from np.append()!
        else:
            clusters = KMeans(num).fit(region.coords)
            centroids = np.round(clusters.cluster_centers_)
            for centroid in centroids:
                centroid = list(centroid)
            wildebeest.append(centroid)

    del regions
    points = []
    for point in wildebeest:
        #rows, cols = zip(*centroid)
        x,y = rasterio.transform.xy(transform, point[0], point[1])
        point = Point(x, y)
        points.append(point)


    return points


#    Adapetd from code by Sizhuo Li, University of Copenhagen, and Ankit Kariryaa, University of Bremen

# Methods to add results of a patch to the total results of a larger area. The operator could be min (useful if there are too many false positives), max (useful for tackle false negatives)
def addTOResult(res, prediction, row, col, he, wi, operator = 'MAX'):
    currValue = res[row:row+he, col:col+wi]
    newPredictions = prediction[:he, :wi]
# IMPORTANT: MIN can't be used as long as the mask is initialed with 0!!!!! If you want to use MIN initial the mask with -1 and handle the case of default value(-1) separately.
    if operator == 'MIN': # Takes the min of current prediction and new prediction for each pixel
        currValue [currValue == -1] = 1 #Replace -1 with 1 in case of MIN
        resultant = np.minimum(currValue, newPredictions)
    elif operator == 'MAX':
        resultant = np.maximum(currValue, newPredictions)
    else: #operator == 'REPLACE':
        resultant = newPredictions
# Alternative approach; Lets assume that quality of prediction is better in the centre of the image than on the edges
# We use numbers from 1-5 to denote the quality, where 5 is the best and 1 is the worst.In that case, the best result would be to take into quality of prediction based upon position in account
# So for merge with stride of 0.5, for eg. [12345432100000] AND [00000123454321], should be [1234543454321] instead of [1234543214321] that you will currently get.
# However, in case the values are strecthed before hand this problem will be minimized
    res[row:row+he, col:col+wi] =  resultant
    return (res)

# Methods that actually makes the predictions
def predict_using_model(model, weight_path, num_folds, batch, batch_pos, mask, operator):
    tm = np.stack(batch, axis = 0)


    num_fold = num_folds
    Ypredict = [0]

    for i in range(num_fold):
        print(f"Predicting using sub-model {i+1}")

        best_path = os.path.join(weight_path, 'best_weights_fold_' + str(i+1) + '.hdf5')
        # best_path = os.path.join(WEIGHT_PATH, 'best_weights.hdf5')
        model.load_weights(best_path)
        predict = model.predict(tm)
        #print(np.max(predict))
        #print(np.min(predict))
        if np.max(predict) > 0.05:
          predict = (predict-np.min(predict))/(np.max(predict)-np.min(predict))
        # print(np.max(predict))
        # print(np.min(predict))
        # print(np.shape(predict))
        Ypredict += predict
        del predict

    predict = Ypredict/num_fold
    del Ypredict

    # print(np.max(predict))
    # print(np.min(predict))
    predict[predict>0.5]=1.0
    predict[predict<=0.5]=0.0
    predict = predict.astype(int)

    # for i in range(len(predict)):
    #   visualize_data(tm[i].astype(np.uint8), np.expand_dims(predict[i], axis=2))

    for i in range(len(batch_pos)):
        print(f"Adding result of patch {i}...")
        (col, row, wi, he) = batch_pos[i]
        p = np.squeeze(predict[i], axis = -1)
        # del predict
        # Instead of replacing the current values with new values, use the user specified operator (MIN,MAX,REPLACE)
        mask = addTOResult(mask, p, row, col, he, wi, operator)
    return mask


def detect_wildebeest(model, weight_path, src, width=PATCH_SIZE, height=PATCH_SIZE, stride = 128, batch_size=12, stretch=False, num_folds=10):
    num_bands = len(INPUT_BANDS)
    nols, nrows = src.meta['width'], src.meta['height']
    print(f"Dimension of this tile: number of columns {nols} number of rows {nrows}.")
    meta = src.meta.copy()

    # if nols > stride and nrows > stride:
    #   offsets = product(range(0, nols-stride, stride), range(0, nrows-stride, stride))
    # else:
    offsets = product(range(0, nols, stride), range(0, nrows, stride))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    #print(nrows, nols)

    mask = np.zeros((nrows, nols))

#     mask = mask -1 # Note: The initial mask is initialized with -1 instead of zero to handle the MIN case (see addToResult)
    batch = []
    batch_pos = [ ]
    for col_off, row_off in  tqdm(offsets):
        if nols-col_off < width:
          col_off = nols-width
        if nrows-row_off < height:
          row_off = nrows-height
        small_window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        #######print(f"Small window {small_window}")
        transform = windows.transform(small_window, src.transform)
        #print(transform)
        patch = np.zeros((height, width, num_bands)) #Add zero padding in case of corner images
        raster_array = src.read(window=small_window)
        raster_array = np.moveaxis(raster_array, 0, -1)[:, :, INPUT_BANDS]

        if stretch:
            raster_array = linear_stretch(raster_array)
            print("stretched")

        patch[:small_window.height, :small_window.width] = raster_array
        batch.append(patch)
        batch_pos.append((small_window.col_off, small_window.row_off, small_window.width, small_window.height))
        if (len(batch) == batch_size):
            print(np.shape(batch))
            mask = predict_using_model(model, weight_path, num_folds, batch, batch_pos, mask, 'MAX')
            # visualize_prediction(mask)
            batch = []
            batch_pos = []

    # To handle the edge of images as the image size may not be divisible by n complete batches
    # and few frames on the edge may be left.
    if batch:
        print(np.shape(batch))
        mask = predict_using_model(model, weight_path, num_folds, batch, batch_pos, mask, 'MAX')
        batch = []
        batch_pos = []

    return mask

def createShapefileObject(points, crs, wfile):
    schema = {
        'geometry': 'Point',
        'properties': {'id': 'str'},
        }
    #with fiona.open(wfile, 'w', crs=meta.get('crs').to_dict(), driver='ESRI Shapefile', schema=schema) as sink:
    with fiona.open(wfile, 'w', crs=crs, driver='ESRI Shapefile', schema=schema) as sink:
        for idx, point in enumerate(points):
            sink.write({
                'geometry': mapping(point),
                'properties': {'id': str(idx)},
                })
            #print(mapping(point))
def writeResultsToDisk(detected_mask, src, transform, out_path=None, mask_path=None, cluster_size=9):

    if out_path != None:
      all_points = ImageToPoints(detected_mask, src, cluster_size) #the projection of points here follow the projection of the image (4326 for 2023)
      print("Number of detected wildebeest: ", len(all_points))
      count = len(all_points)

      # detected_meta = src.meta.copy()
      createShapefileObject(all_points, src.crs, out_path) #

    if mask_path != None:
      crs = src.meta['crs']
      new_dataset = rasterio.open(mask_path, 'w', driver='GTiff',
                    height = detected_mask.shape[0], width = detected_mask.shape[1],
                    count=1, dtype=str(detected_mask.dtype),
                    crs=crs,
                    transform=transform)
      new_dataset.write(detected_mask[:,:],1)
      new_dataset.close()



def detect_wildebeest_tile(model, weight_path, src, Output_dir, image_dir, tile_width=5000, tile_height=5000, 
                           width=PATCH_SIZE, height=PATCH_SIZE, 
                           stride = 128, batch_size=12, stretch=False, num_folds=10,
                           mask_outpath=None, cluster_size=9):
    
    file_name = os.path.split(image_dir)[1]
    name, file_extension = os.path.splitext(file_name)
    output_fn = name+'_tile_{}_{}'

    meta = src.meta.copy()
    for big_window, transform in get_tiles(src, tile_width, tile_height ):
        #print(window)
        meta['transform'] = transform
        meta['width'], meta['height'] = big_window.width, big_window.height
        
        out_name = output_fn.format(int(big_window.col_off), int(big_window.row_off))
        
        shp_path = os.path.join(Output_dir, out_name+'.shp')
        mask_path = os.path.join(Output_dir, out_name+'.tif')

        if Path(shp_path).is_file() == True:
          print(f"Prediction for tile {out_name} already exists. Skip.")
          continue
           
  
        num_bands = len(INPUT_BANDS)
        nols, nrows = meta['width'], meta['height'] 
        print(f"Dimension of this tile: number of columns {nols} number of rows {nrows}.")
        # meta = src.meta.copy()

    # if nols > stride and nrows > stride:
    #   offsets = product(range(0, nols-stride, stride), range(0, nrows-stride, stride))
    # else:
        offsets = product(range(0, nols, stride), range(0, nrows, stride))
        #print("tile window: ", big_window)

        mask = np.zeros((nrows, nols))
    
    #     mask = mask -1 # Note: The initial mask is initialized with -1 instead of zero to handle the MIN case (see addToResult)
        batch = []
        batch_pos = [ ]
        for col_off, row_off in  tqdm(offsets):
            if nols-col_off < width:
              col_off = nols-width
            if nrows-row_off < height:
              row_off = nrows-height
            small_window =windows.Window(col_off=col_off+big_window.col_off, row_off=row_off+big_window.row_off, width=width, height=height).intersection(big_window)
            patch = np.zeros((height, width, num_bands)) #Add zero padding in case of corner images
            raster_array = src.read(window=small_window)
            raster_array = np.moveaxis(raster_array, 0, -1)[:, :, INPUT_BANDS]
    
            if stretch:
                raster_array = linear_stretch(raster_array)
                print("stretched")
    
            patch[:small_window.height, :small_window.width] = raster_array
            batch.append(patch)
            batch_pos.append((col_off, row_off, small_window.width, small_window.height))
            if (len(batch) == batch_size):
                print(np.shape(batch))
                mask = predict_using_model(model, weight_path, num_folds, batch, batch_pos, mask, 'MAX')
                # visualize_prediction(mask)
                batch = []
                batch_pos = []
    
        # To handle the edge of images as the image size may not be divisible by n complete batches
        # and few frames on the edge may be left.
        if batch:
            print(np.shape(batch))
            mask = predict_using_model(model, weight_path, num_folds, batch, batch_pos, mask, 'MAX')
            batch = []
            batch_pos = []
            
        if mask_outpath == None:
            writeResultsToDisk(mask, src, transform, shp_path, mask_outpath, cluster_size)
        else:
            writeResultsToDisk(mask, src, transform, shp_path, mask_path, cluster_size)
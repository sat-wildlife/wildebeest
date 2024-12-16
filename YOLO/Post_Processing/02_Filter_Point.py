import os
import fnmatch
from osgeo import gdal, ogr, osr
import numpy as np
from scipy.ndimage import maximum_filter
from tqdm import tqdm

# Define directories for input and output data
directory = r""  # Directory containing raster (.tif) files
shp_directory = r""  # Directory containing input shapefiles
shp_out_directory = r""  # Directory to save processed output shapefiles

# Create the output directory if it doesn't exist
if not os.path.exists(shp_out_directory):
    os.makedirs(shp_out_directory)

# Define the file pattern to match raster files
pattern = '*.tif'

# Iterate through all files in the input directory
for file in os.listdir(directory):
    if fnmatch.fnmatch(file, pattern):  # Check if the file matches the pattern
        raster_path = os.path.join(directory, file)  # Full path to the raster file
        shapefile_path = os.path.join(shp_directory, '***_' + file[:-4] + '.shp')  # Corresponding input shapefile path
        output_shapefile_path = os.path.join(shp_out_directory, file[:-4] + '.shp')  # Path for the output shapefile

        # Open the raster file and extract its properties
        raster_ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
        geo_transform = raster_ds.GetGeoTransform()  # Get the raster's geotransform
        raster_band = raster_ds.GetRasterBand(1)  # Access the first band of the raster
        xsize = raster_band.XSize  # Number of columns in the raster
        ysize = raster_band.YSize  # Number of rows in the raster

        # Set the block size for processing the raster
        block_size = 1024

        # Initialize global maps for point counts and confidence values
        global_points_map = np.zeros((ysize, xsize), dtype=np.int)  # Map to count points
        global_conf_map = np.full((ysize, xsize), fill_value=-np.inf, dtype=np.float16)  # Map for confidence values

        # Open the input shapefile for reading
        driver = ogr.GetDriverByName('ESRI Shapefile')
        vector_ds = driver.Open(shapefile_path, 0)
        layer = vector_ds.GetLayer()

        # Create the output shapefile and define its structure
        new_ds = driver.CreateDataSource(output_shapefile_path)
        srs = osr.SpatialReference(wkt=raster_ds.GetProjection())  # Spatial reference from the raster
        new_layer = new_ds.CreateLayer('', srs, ogr.wkbPoint)  # Create a point layer
        conf_field = ogr.FieldDefn("Conf", ogr.OFTReal)  # Add a field for confidence values
        new_layer.CreateField(conf_field)
        layer_defn = new_layer.GetLayerDefn()

        # Process the raster in blocks
        for y in tqdm(range(0, ysize, block_size), desc="Processing blocks"):
            for x in range(0, xsize, block_size):
                block_xsize = min(block_size, xsize - x)  # Determine the block width
                block_ysize = min(block_size, ysize - y)  # Determine the block height

                # Initialize local maps for the current block
                points_map = np.zeros((block_ysize, block_xsize), dtype=np.int)
                conf_map = np.full((block_ysize, block_xsize), fill_value=-np.inf, dtype=np.float32)

                # Reset the layer and iterate through its features
                layer.ResetReading()
                for feature in layer:
                    geom = feature.GetGeometryRef()
                    x_point, y_point = geom.GetX(), geom.GetY()  # Get the point's coordinates
                    conf = feature.GetField("Conf")  # Get the confidence value
                    col, row = map(int, [(x_point - geo_transform[0]) / geo_transform[1] - x,
                                         (y_point - geo_transform[3]) / geo_transform[5] - y])  # Map point to raster grid
                    if 0 <= row < block_ysize and 0 <= col < block_xsize:  # Check if the point falls within the block
                        points_map[row, col] += 1  # Increment the point count
                        conf_map[row, col] = max(conf_map[row, col], conf)  # Update confidence value if higher

                # Apply a maximum filter to find local maxima
                footprint = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # Define the neighborhood
                max_neighbor_count = maximum_filter(points_map, footprint=footprint, mode='constant', cval=0)
                max_neighbor_conf = maximum_filter(conf_map, footprint=footprint, mode='constant', cval=-np.inf)
                mask = (points_map > max_neighbor_count) | (
                        (points_map == max_neighbor_count) & (conf_map > max_neighbor_conf))  # Identify local maxima
                local_y, local_x = np.where(mask)  # Get the coordinates of local maxima

                # Update global maps with the current block's data
                global_points_map[y:y + block_ysize, x:x + block_xsize] = points_map
                global_conf_map[y:y + block_ysize, x:x + block_xsize] = conf_map

                # Write the local maxima points to the output shapefile
                for i in range(len(local_x)):
                    global_x = x + local_x[i]
                    global_y = y + local_y[i]
                    center_x = geo_transform[0] + global_x * geo_transform[1] + geo_transform[1] / 2  # Calculate X coordinate
                    center_y = geo_transform[3] + global_y * geo_transform[5] + geo_transform[5] / 2  # Calculate Y coordinate
                    selected_conf = conf_map[local_y[i], local_x[i]]  # Get the confidence value

                    point = ogr.Geometry(ogr.wkbPoint)  # Create a point geometry
                    point.AddPoint(center_x, center_y)
                    feature = ogr.Feature(layer_defn)
                    feature.SetGeometry(point)  # Set the geometry of the feature
                    feature.SetField("Conf", float(selected_conf))  # Set the confidence field
                    new_layer.CreateFeature(feature)  # Add the feature to the layer

        new_ds = None  # Close the output shapefile

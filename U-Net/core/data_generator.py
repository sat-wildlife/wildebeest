import tensorflow as tf
import numpy as np
import rasterio
import cv2

import os
from osgeo import gdal
import rasterio
from rasterio import windows


import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
'''
Before training, we split the data into training and validation folds, get train_df and val_df for this specific sub-model

In data generator, we :

- read a patched image and patched label using the window info, 
- add zero padding if window is smaller than patchsize*patchsize, 
- load in batch, 
- run augmentation, 
- feed to fit

Train, validate

For testing, we read the patched image and patched label, read the window and get the transform. Predict and then evaluate using the transform.
But this way is the last window is smaller than a patch, how will it be evaluated?
image -> add zero padding if size is smaller
after prediction -> cut using the actual window
mask -> keep the same: actual window size
'''

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_IDs, batch_size=2, patchsize=336, input_image_channel=[0,1,2],
                 shuffle=True, augment=True, folder="/home/zijing/wildebeest/SampleData", weight_type = None, weight=None):
        'Initialization'
        self.patchsize = patchsize
        self.dim = (patchsize, patchsize)
        self.batch_size = batch_size
        self.image_IDs = image_IDs

        self.input_image_channel = input_image_channel
        self.n_channels = len(input_image_channel)

        self.shuffle = shuffle
        self.augment = augment
        self.folder = folder
        self.weight = weight
        self.weight_type = weight_type
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        '''
        The __getitem__ function taking a batch of data from the dataframe using indexing and passing it into the __get_data function to get X and y to be used for training. 
        The index passed into the function will be done by the fit function while training. Select the correct batch of data by using this index.
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image_IDs_temp = self.image_IDs.loc[indexes].reset_index(drop=True)

        # Generate data
        X, Y, sw = self.__data_generation(image_IDs_temp)

        return tf.convert_to_tensor(X, dtype=np.float32), tf.convert_to_tensor(Y,dtype=np.float32),tf.convert_to_tensor(sw,dtype=np.float32)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, image_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8) # zero padding in case the image is smaller than patch size
        Y = np.zeros((self.batch_size, *self.dim, 1), dtype=np.int32)

        sample_weight = np.ones((self.batch_size, *self.dim, 1))
            
        # Generate data
        for i, row in image_IDs_temp.iterrows():
            # Store sample
                
            if row['FileName'].endswith("tif"):
                image_path = os.path.join(self.folder, row['FileName'])
                label_path = os.path.join(self.folder, row['LabelName'])
                small_window = windows.Window(col_off=row['Window_col_off'], row_off=row['Window_row_off'], 
                                              width=row['Window_width'], height=row['Window_height'])
                image_channel = [int(i)-1 for i in str(row['bandorder'])]
                actual_image_channel = [image_channel[k] for k in self.input_image_channel]
                # if row['Stretch'] == 0:
                Xi = np.moveaxis(rasterio.open(image_path).read(window=small_window), 0, -1)[:, :, actual_image_channel]
                # if row['Stretch'] == 1:
                    # Xi = linear_stretch(np.moveaxis(rasterio.open(image_path).read(window=small_window), 0, -1)[:, :, actual_image_channel])
                X[i,:small_window.height,:small_window.width, :] = Xi[:,:,self.input_image_channel]
    #             X = X/255
                # Store class
                label = np.moveaxis(rasterio.open(label_path).read(window=small_window), 
                                                                               0, -1)
                Y[i,:label.shape[0],:label.shape[1], :] = label[:, :,:]

            else:
                #if images are already cropped to patchsize*patchsize and saved to jpg or other format 
                image_path = os.path.join(self.folder, row['FileName'])
                label_path = os.path.join(self.folder, row['LabelName'])
    
                actual_image_channel = [2,1,0]
                # if row['Stretch'] == 0:
                Xi = cv2.imread(image_path)[:, :, actual_image_channel]
                
                X[i,:Xi.shape[0],:Xi.shape[1],:] = Xi[:,:,:]
    #             X = X/255
                # Store class
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                
                Y[i,:label.shape[0],:label.shape[1], 0] = label[:, :]
            if self.weight != None:
                if row['DataYear'] == self.weight_type:
                    sample_weight[i,:,:] == self.weight
                    

        if self.augment== True:   
            seq = iaa.Sequential([
                #iaa.Dropout([0.05, 0.2]),# drop 5% or 20% of all pixels
                iaa.Flipud(0.5), #affects segmaps
                iaa.Fliplr(0.5), #affects segmaps
                iaa.geometric.Rot90(ia.ALL), #affects segmaps
                iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
                iaa.LinearContrast((0.4,1.6)),
                iaa.Affine(scale=(0.8, 1.2))
              # iaa.Sharpen((0.4, 0.6))      # sharpen the image
              #iaa.Affine(rotate=(-90, 90))  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)
            
            ia.seed(1)
            
            # Augment images and segmaps.
            images_aug = []
            segmaps_aug = []
            for i in range(len(image_IDs_temp)):
              segmap = SegmentationMapsOnImage(Y[i], shape=X[i].shape)
              images_aug_i, segmaps_aug_i = seq(image=X[i], segmentation_maps=segmap)
              images_aug.append(images_aug_i)
              segmaps_aug.append(segmaps_aug_i.get_arr())

            return np.array(images_aug), np.array(segmaps_aug), sample_weight
        else:
            return X, Y, sample_weight
            


class SimpleDataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, image_IDs, batch_size=2, patchsize=336, input_image_channel=[0,1,2,3],
                 shuffle=True, augment=True):
        'Initialization'
        self.patchsize = patchsize
        self.dim = (patchsize, patchsize)
        self.batch_size = batch_size
        self.image_IDs = image_IDs
        # self.image_IDs = image_IDs[(image_IDs['Window_width']==336) & (image_IDs['Window_height']==336)].reset_index(drop=True)
        #self.label_IDs = label_IDs
        self.input_image_channel = input_image_channel
        self.n_channels = len(input_image_channel)

        #self.meta_list = self.load_meta(self.image_IDs, patchsize)
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        '''
        The __getitem__ function taking a batch of data from the dataframe using indexing and passing it into the __get_data function to get X and y to be used for training. 
        The index passed into the function will be done by the fit function while training. Select the correct batch of data by using this index.
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        image_IDs_temp = self.image_IDs.loc[indexes].reset_index(drop=True)

        # Generate data
        X, Y = self.__data_generation(image_IDs_temp)

        return tf.convert_to_tensor(X, dtype=np.float32), tf.convert_to_tensor(Y,dtype=np.float32) 

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, image_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels), dtype=np.uint8) # zero padding in case the image is smaller than patch size
        Y = np.zeros((self.batch_size, *self.dim, 1), dtype=np.int32)

        # Generate data
        for i, row in image_IDs_temp.iterrows():
            # Store sample
            image_path = row['FileName']
            label_path = row['LabelName']

            actual_image_channel = [2,1,0]
            # if row['Stretch'] == 0:
            Xi = cv2.imread(image_path)[:, :, actual_image_channel]
                # Xi = np.moveaxis(rasterio.open(image_path).read(window=small_window), 0, -1)[:, :, actual_image_channel]
            
            X[i,:Xi.shape[0],:Xi.shape[1],:] = Xi[:,:,:]
#             X = X/255
            # Store class
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            
            Y[i,:label.shape[0],:label.shape[1], 0] = label[:, :]

        if self.augment== True:   
            seq = iaa.Sequential([
              #iaa.Dropout([0.05, 0.2]),# drop 5% or 20% of all pixels
              iaa.Flipud(0.5), #affects segmaps
              iaa.Fliplr(0.5), #affects segmaps
              iaa.geometric.Rot90(ia.ALL), #affects segmaps
              iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
              iaa.LinearContrast((0.4,1.6)),
              # iaa.Sharpen((0.4, 0.6))      # sharpen the image
              iaa.Affine(scale=(0.8, 1.2))  # rotate by -45 to 45 degrees (affects segmaps)
            ], random_order=True)
            
            ia.seed(1)
            
            # Augment images and segmaps.
            images_aug = []
            segmaps_aug = []
            for i in range(len(image_IDs_temp)):
              segmap = SegmentationMapsOnImage(Y[i], shape=X[i].shape)
              images_aug_i, segmaps_aug_i = seq(image=X[i], segmentation_maps=segmap)
              images_aug.append(images_aug_i)
              segmaps_aug.append(segmaps_aug_i.get_arr())
            
            return np.array(images_aug), np.array(segmaps_aug)
        else:
            return X, Y
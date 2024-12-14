#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import rasterio
from rasterio import windows
from rasterio.windows import Window
import pandas as pd
import glob
from itertools import product


PATCH_SIZE = 336
def prep_directory(Data_folder, year, image_path, label_path, out_path, bandorder="1234", stretch=0, stride=PATCH_SIZE, ext="tif"):
    '''
    head: "data_2009Aug"
    stride: 
    year_list = ["data_2009Aug", "data_2010Sep", "data_2013Aug", "data_2015Jul", "data_2018Aug", "data_2020Oct"]
    image_path: folder where you store the images, e.g. Data_folder+'/'+year+'/'+'3_Train_test/train/image_match2023_4'
    label_path: folder where you store the masks, e.g. Data_folder+'/'+year+'/'+'3_Train_test/train/mask'
    stretch: 0 or 1
    bandorder: "1234" or "3214"
    out_path: path to save the data directory, csv format, e.g. os.path.join(Data_folder, 'update_match2023_4_dict_train_filelinks.csv')
    '''
    data_df = pd.DataFrame(columns=['FileName', 'LabelName', 'DataYear','PatchSize', 'Stretch',
                                    'Window_col_off','Window_row_off','Window_width','Window_height'])
    
    data_dict = {}
    data_dict['FileName']=[]
    data_dict['LabelName']=[]
    data_dict['DataYear']=[]
    data_dict['PatchSize']=[]
    data_dict['Stretch']=[]
    data_dict['bandorder']=[]
    data_dict['Window_col_off']=[]
    data_dict['Window_row_off']=[]
    data_dict['Window_width']=[]
    data_dict['Window_height']=[]
     
    
    dir_image_list = glob.glob(image_path+"/"+"*.{}".format(ext))
    dir_label_list = glob.glob(label_path+"/"+"*.{}".format(ext))
        
    for i in range(len(dir_image_list)):
        image_dir = dir_image_list[i]
        label_dir = dir_label_list[i]
    
        image_rel = os.path.relpath(image_dir, Data_folder)
        label_rel = os.path.relpath(label_dir, Data_folder)
    #         print(image_dir)
        src = rasterio.open(image_dir)
        ncols, nrows = src.meta['width'], src.meta['height']
    #         print(f"Dimension of this tile: number of columns {ncols} number of rows {nrows}.")
        
        src2 = rasterio.open(label_dir)
        ncols2, nrows2 = src2.meta['width'], src2.meta['height'] 
        if (ncols != ncols2) or (nrows != nrows2):
            print('Image and mask are NOT in the same dimension!')
            print(image_dir)
            print(label_dir)
            print(ncols, nrows)
            print(ncols2, nrows2)
        
        offsets = product(range(0,ncols,stride), range(0,nrows,stride))
        big_window = windows.Window(0,0,ncols,nrows)
        
        for col_off, row_off in offsets:
            if ncols-col_off >= PATCH_SIZE * 0.25 and ncols-col_off < PATCH_SIZE:
                col_off = ncols-PATCH_SIZE
            if nrows-row_off >= PATCH_SIZE * 0.25 and nrows-row_off < PATCH_SIZE:
                row_off = nrows-PATCH_SIZE
            small_window = windows.Window(col_off, row_off, PATCH_SIZE, PATCH_SIZE).intersection(big_window)
            w_col_off, w_row_off, w_width, w_height = small_window.col_off, small_window.row_off, small_window.width, small_window.height
            
            data_dict['FileName'].append(image_rel)
            data_dict['LabelName'].append(label_rel)
            data_dict['DataYear'].append(year)
            data_dict['Stretch'].append(stretch)
            data_dict['bandorder'].append(bandorder)
            data_dict['PatchSize'].append(PATCH_SIZE)
            data_dict['Window_col_off'].append(w_col_off)
            data_dict['Window_row_off'].append(w_row_off)
            data_dict['Window_width'].append(w_width)
            data_dict['Window_height'].append(w_height)            
                        
    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(out_path)



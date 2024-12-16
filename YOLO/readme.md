
Use YOLO to detect wildebeest (and zebras) and get point-level predictions in the Serengeti-Mara ecosystem from very-high-resolution satellite imagery. 
Please feel free to contact me at xuzeyu689@gmail.com if you have any questions.

# Setup and Installation

# Steps
## Step 1: Data preparation - [Data_Preparation]
- **01_YOLO_Samples.py**: Generates sample path files in YOLO format.  
- **02_Random Image Crop.py**: Randomly extracts fixed-size sub-images from large images for histogram matching.  
- **03_Image_Stitching.py**: Stitches the sub-images from the previous step into a single large image.  
- **04_Histogram Matching.py**: Performs histogram matching for the samples.  

## Step 2: Model training - [YOLO_Model]
Run train.py

## Step 3: Detection on new satellite imagery - [YOLO_Model]

We implemented the YOLO v8 code based on Ultralytics and modified the predictor to enable the model to be directly applied to large satellite imagery.

Before running, modify the file paths in ultralytics/engine/predictor.py. Next, adjust the paths in predict.py and execute the script.

## Step 4: Data - [Post_Processing]

- **01_Object_Center.py**: Extracts the center points of bounding boxes as wildebeest target points. Both input and output are in Esri Shapefile format.  
- **02_Filter_Point.py**: Filters target points based on the image, retaining only one target point within a single pixel range.  
- **03_Shapefile_Clipping.py**: Clips Shapefile vectors.
- 
## Step 3: Accuracy assessment - [YOLO_Model]

This part is consistent with the corresponding part in our U-Net implementation.

## Acknowledgement
The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics).
Appreciate the excellent implementations!

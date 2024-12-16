
Use YOLO to detect wildebeest (and zebras) and get point-level predictions in the Serengeti-Mara ecosystem from very-high-resolution satellite imagery. 
Please feel free to contact me at xuzeyu689@gmail.com if you have any questions.

# Setup and Installation

# Steps
## Step 1: Data preparation - [Data_Preparation]
- **01_YOLO_Samples.py**: Used to generate sample path files in YOLO format.  
- **02_Random Image Crop.py**: Used to randomly extract fixed-size sub-images from large images for histogram matching.  
- **03_Image_Stitching.py**: Used to stitch the sub-images from the previous step into a single large image.  
- **04_Histogram Matching.py**: Used for histogram matching of the samples.  
## Step 2: Model training - [2_Model training.ipynb]
Prepare the data directories from the dataset above.
Train the U-Net-based model using the ensemble approach.
## Step 3: Accuracy assessment - [3_Test accuracy.ipynb]
Calculate the accuracy of the model on test dataset at the wildebeest-individual level.
## Step 4: Detection on new satellite imagery - [4_Prediction_Detection wildebeest.ipynb]
Run the model on new satellite imagery to get point-level predictions.
## Acknowledgement
The code base is built with [ultralytics](https://github.com/ultralytics/ultralytics).
Thanks for the great implementations!

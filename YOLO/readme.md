
Use YOLO to detect wildebeest (and zebras) and get point-level predictions in the Serengeti-Mara ecosystem from very-high-resolution satellite imagery. 
Please feel free to contact me at xuzeyu689@gmail.com if you have any questions.

# Setup and Installation

# Steps
## Step 1: Data preparation - [1_Prepare image and masks.ipynb]
Crop the image patches from the satellite image according to the areas of interest.</p>
Create the segmentation masks from the point annotations of animal individuals.</p>
Please note that the commercial satellite imagery dataset used in this study is not publicly available due to copyright issues. Please use your own data instead.
## Step 2: Model training - [2_Model training.ipynb]
Prepare the data directories from the dataset above.
Train the U-Net-based model using the ensemble approach.
## Step 3: Accuracy assessment - [3_Test accuracy.ipynb]
Calculate the accuracy of the model on test dataset at the wildebeest-individual level.
## Step 4: Detection on new satellite imagery - [4_Prediction_Detection wildebeest.ipynb]
Run the model on new satellite imagery to get point-level predictions.
# References
For more technical details please refer to our paper (https://www.nature.com/articles/s41467-023-38901-y).

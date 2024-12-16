
Use the segmentation-based neural network model (UNet) to detect wildebeest (and zebras) and get point-level predictions in the Serengeti-Mara ecosystem from very-high-resolution satellite imagery. 
Please feel free to contact me at zijingwu97@outlook.com if you have any questions.
# Setup and Installation
The easiest way to test and use this code is to upload all the files to Google Drive and open the notebooks with Google Colaboratory (https://colab.research.google.com/). </p>
Alternatively, you can install the required packages (see tensorflow_environment.yml) on your computer and use the notebooks with Jupyter Notebook. </p>
Note: If you use Google Colaboratory, you may encounter some issues with deprecated arguments/functions because Colaboratory keeps updating the packages. These issues are usually easy to solve though, so don't hesitate to give it a try!
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

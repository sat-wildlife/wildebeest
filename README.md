# Deep learning-based satellite survey of wildebeest in Serengeti-Mara ecosystem
This repository contains the code for detecting wildebeest in Serengeti-Mara ecosystem from very-high-resolution satellite imagery.</p>
In this study, we conducted satellite surveys across northern Serengeti National Park in Tanzania and southwestern Kenya, 
encompassing the Masai Mara National Reserve over two consecutive years. 
The image resolutions range from 33 to 60 cm, capturing each wildebeest as a 6- to 12-pixel representation. 
We utilized two deep-learning models to automate the detection and counting process: U-Net, a pixel-based segmentation model, and YOLO, an object-based detection model.
Applying both models allowed us to compare their efficacy in population estimation within this large-scale wildlife survey.
## U-Net
U-Net is tailored for image segmentation tasks. It operates by assigning each pixel within an image to a class label, 
allowing for precise localization and identification of wildebeest pixels. We then extract the location of individual wildebeest as points from the wildebeest segmentation maps.

## YOLO
YOLO, or 'You Only Look Once', is optimized for real-time object detection. 
This model identifies bounding boxes around objects, which enables the detection of wildebeest as discrete entities within the imagery. 
The code base of YOLO is built with [ultralytics](https://github.com/ultralytics/ultralytics). Appreciate the excellent implementations!

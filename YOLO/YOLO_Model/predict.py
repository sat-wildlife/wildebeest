import cv2
import numpy as np
from ultralytics import YOLO

# We modified the prediction part of the model, so the configuration for this part is relatively simple.

if __name__ == '__main__':

    test_img='D:/test_model.png'

    model = YOLO('****.pt')

    results = model(test_img,imgsz=512,conf=0.1)

if __name__ == '__main__':
    from ultralytics import YOLO

    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    model = YOLO('ultralytics/models/yolov8x.yaml').load('***yolov8x.pt')

    # Train the model
    model.train(data='ultralytics/data/wdata.yaml', batch=5, epochs=1000, imgsz=512,workers=1)



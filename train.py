from ultralytics import YOLO

if __name__=='__main__':
    model = YOLO("yolov8m.yaml")    # Train from scratch
    model.train(data='../data/cell_img_dataset_metadata.yaml',
                epochs = 5000,
                patience = 500,         # 0 to disable
                conf = None,
                iou = 0.7,
                lr0 = 0.01,             # Initial learning rate
                lrf = 0.01,              # Final learning rate
                hsv_h = 1.0,          # HSV-Hue augmentation
                hsv_s = 1.0,          # HSV-Saturation augmentation
                hsv_v = 0.7,          # HSV-Value (brightness) augmentation
                degrees = 45.0,       # Image rotation (+/- deg)
                translate = 0.1,      # Image translation (+/- fraction)
                scale = 0.3,          # Image scale (+/- gain)
                shear = 0.0,          # Image shear (+/- deg)
                perspective = 0.0,    # Image perspective (+/- fraction)
                flipud = 0.5,         # Image flip up-down (probability)
                fliplr = 0.5,         # Image flip left-right (probability)
                mixup = 0.0,
                batch = 16,
                visualize = False,
                imgsz = 640,
                device = 0,
                workers = 1)
import numpy as np
import cv2
import torch
import os
from ultralytics import YOLO

class CNN_Detector():
    """
    YOLOv8 Deep Convolutional Neural Network object detection class

    Attributes:
    - device (torch.device): The device (CPU or GPU) used for inference.
    - conf_thresh (float): Confidence threshold for detections.
    - iou_thresh (float): IoU threshold for NMS.
    - model (torch.nn.Module): The pre-trained detection model.
  
    Methods:
    - detect(input_img): Perform object detection on an input image.
    - letterbox(img, new_shape, color, auto, scaleFill, scaleup, stride): Resize and pad image while
      meeting stride-multiple constraints.
    """
    def __init__(self, weights_fname, inference_device, image_size=640, confidence_thresh=0.5, iou_thresh = 0.45):
        """
        Initializes the CNN_Detector object.
        Args:
            weights_fname (str): The name of the file containing the pre-trained model's weights.
            inference_device (str): Device to use for model inference
            image_size (int, optional): The size of the input images.
            confidence_thresh (float, optional): The minimum confidence threshold for object detection. Default is 0.5.
            iou_thresh (float, optional): The minimum intersection over union (IoU) threshold for object detection. Default is 0.45.
        """
        # Initialise
        self.device = self.select_device(inference_device)
        self.conf_thresh = confidence_thresh
        self.iou_thresh = iou_thresh
        self.image_size = image_size

        # Load model
        self.model = YOLO(weights_fname)

        # Warmup if using GPU
        if self.device.type != 'cpu':
            self.model.predict(np.zeros((image_size,image_size,3)), verbose=False)

    def detect(self, input_img):
        """
        Detects objects in the input image.
        Args:
            input_img (numpy.ndarray): The input image.
        Returns:
            List: A List containing class_enum, confidence, and bounding box x1, y1, x2, y2 for each detection
        """
        img = np.ascontiguousarray(input_img)
        img = img.astype(float) / 255.0  # 0 - 255 to 0.0 - 1.0

        # Inference
        with torch.no_grad():
            results = self.model.predict(input_img,
                                            conf=self.conf_thresh,
                                            iou=self.iou_thresh,     # IoU threshold for NMS
                                            agnostic_nms=True,      # Class agnostic NMS cos we don't want multiple classifications on one cell
                                            max_det=300,       # Max number of detections (per tile)
                                            verbose=False)

        detections = []
        for res in results[0].boxes:  # detections per image
            detection = np.r_[res.cls.cpu().numpy(), res.conf.cpu().numpy(), np.squeeze(res.xyxy.cpu().numpy())]     # [class_enum, confidence, x1, y1, x2, y2]
            detections.append(detection)

        return detections

    def select_device(self, device='', batch_size=None):
        s = ''
        cpu = device.lower() == 'cpu'
        if cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
        elif device:  # non-cpu device requested
            os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
            assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

        cuda = not cpu and torch.cuda.is_available()
        if cuda:
            n = torch.cuda.device_count()
            if n > 1 and batch_size:  # check that batch_size is compatible with device_count
                assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
            space = ' ' * len(s)
            for i, d in enumerate(device.split(',') if device else range(n)):
                p = torch.cuda.get_device_properties(i)
                s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
        else:
            s += 'CPU\n'

        #logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
        return torch.device('cuda:0' if cuda else 'cpu')
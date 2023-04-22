import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from yolov7_utils.general import set_logging, check_img_size, non_max_suppression
from yolov7_utils.torch_utils import select_device

# Load settings
import yaml
with open('settings.yaml', 'r') as stream:
    try:
        settings = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print(e)

class CNN_Detector():
    def __init__(self, weights_fname, image_size, confidence_thresh=0.5, iou_thresh = 0.45):
        # Initialize
        set_logging()
        self.device = select_device(settings['INFERENCE_DEVICE'])
        self.conf_thresh = confidence_thresh
        self.iou_thresh = iou_thresh

        # Load model
        self.model = attempt_load(weights_fname, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.image_size = check_img_size(image_size, s=self.stride)  # check requested image_size

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()  # to FP16

        # Warmup if using GPU
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, image_size, image_size).to(self.device).type_as(next(self.model.parameters())))  # run once

    def detect(self, input_img):
        # Pre-process image
        img = self.letterbox(input_img, self.image_size, stride=self.stride)[0].copy()  # Padded resize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, re-order to DxHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh)

        for det in pred:  # detections per image
            # Transform box coordinates from processed image frame to original image frame
            bb_coords = self.scale_coords(img.shape[2:], det[:, :4], input_img.shape).round()
            if self.device.type != 'cpu':
                bb_coords = bb_coords.cpu()
            
        results = torch.cat(pred, dim=0).cpu().numpy()[:,::-1]
        # Replace the coordinates in the results array with scaled coordinates
        results[:, 2:] = bb_coords
        
        return results

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clamp_(0, img_shape[1])  # x1
        boxes[:, 1].clamp_(0, img_shape[0])  # y1
        boxes[:, 2].clamp_(0, img_shape[1])  # x2
        boxes[:, 3].clamp_(0, img_shape[0])  # y2

import cv2
import torch
import numpy as np
import supervision as sv
import onnxruntime as ort
import ultralytics.utils.ops as ops

from ultralytics.engine.results import Results
from skimage.measure import approximate_polygon, find_contours

class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model='yolo11n-seg.onnx')
        self.labels = labels
        self.input_size = (640, 640)
        self.conf = 0.25
        self.iou = 0.7

    def load_network(self, model):
        device = ort.get_device()
        try:
            providers = ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_initiated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, img, new_shape=(640, 640)):
        """
        Resizes and pads image while maintaining aspect ratio.

        Args:
            img (np.ndarray): Input image in BGR format.
            new_shape (tuple): Target shape as (height, width).

        Returns:
            (np.ndarray): Resized and padded image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img

    def preprocess(self, img, new_shape):
        """
        Preprocesses the input image before feeding it into the model.

        Args:
            img (np.ndarray): The input image in BGR format.
            new_shape (tuple): The target shape for resizing as (height, width).

        Returns:
            (np.ndarray): Preprocessed image ready for model inference, with shape (1, 3, height, width) and normalized.
        """
        img = self.letterbox(img, new_shape)
        img = img[..., ::-1].transpose([2, 0, 1])[None]  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255  # Normalize to [0, 1]
        return img
    
    def process_mask(self, protos, masks_in, bboxes, shape):

        c, mh, mw = protos.shape  # CHW
        masks = (masks_in @ protos.float().view(c, -1)).view(-1, mh, mw)  # Matrix multiplication
        masks = ops.scale_masks(masks[None], shape)[0]  # Scale masks to original image size
        masks = ops.crop_mask(masks, bboxes)  # Crop masks to bounding boxes
        return masks.gt_(0.0)  # Convert to binary masks

    def postprocess(self, img, prep_img, outs):
        """
        Post-processes model predictions to extract meaningful results.

        Args:
            img (np.ndarray): The original input image.
            prep_img (np.ndarray): The preprocessed image used for inference.
            outs (List): Model outputs containing predictions and prototype masks.

        Returns:
            (List[Results]): Processed detection results containing bounding boxes and segmentation masks.
        """
        preds, protos = [torch.from_numpy(p) for p in outs]
        #preds, protos = [p for p in outs]

        preds = ops.non_max_suppression(preds, self.conf, self.iou, nc=len(self.labels))
        
        print(preds)
        print(protos.shape)
        
        results = []
        for i, pred in enumerate(preds):
            pred[:, :4] = ops.scale_boxes(prep_img.shape[2:], pred[:, :4], img.shape)
            masks = self.process_mask(protos[i], pred[:, 6:], pred[:, :4], img.shape[:2])
            results.append(Results(img, path="", names=self.labels, boxes=pred[:, :6], masks=masks))

        return results
    
    def to_cvat_mask(self, box: list, mask):
        xtl, ytl, xbr, ybr = box
        flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
        flattened.extend([xtl, ytl, xbr, ybr])
        return flattened
    
    def infer(self, image, threshold):
        prep_img = self.preprocess(image, new_shape = self.input_size)
        outs = self.model.run(None, {self.model.get_inputs()[0].name: prep_img})
        result = self.postprocess(image, prep_img, outs)
        
        detections = sv.Detections.from_ultralytics(result[0])
        detections = detections[detections.confidence > threshold]
        
        results=[]
        if len(detections) > 0:
            for xyxy, mask, confidence, class_id, _, _ in detections:
                mask = mask.astype(np.uint8)

                xtl = int(xyxy[0])
                ytl = int(xyxy[1])
                xbr = int(xyxy[2])
                ybr = int(xyxy[3])

                label = int(class_id)
                cvat_mask = self.to_cvat_mask((xtl, ytl, xbr, ybr), mask)

                contours = find_contours(mask, 0.5)
                contour = contours[0]
                contour = np.flip(contour, axis=1)
                polygons = approximate_polygon(contour, tolerance=2.5)

                results.append({
                    "confidence": str(confidence),
                    "label": self.labels.get(class_id, "unknown"),
                    "type": "mask",
                    "points": polygons.ravel().tolist(),
                    "mask": cvat_mask,
                })
        return results
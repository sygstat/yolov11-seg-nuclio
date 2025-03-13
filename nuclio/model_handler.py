import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
import base64


class ModelHandler:
    def __init__(self, labels):
        self.model = None
        self.load_network(model="yolo11n-seg.onnx")
        self.labels = labels
        self.input_size = (640, 640)

    def load_network(self, model):
        device = ort.get_device()
        #cuda = True if device == 'GPU' else False
        try:
            providers = ['CPUExecutionProvider'] #['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
            so = ort.SessionOptions()
            so.log_severity_level = 3

            self.model = ort.InferenceSession(model, providers=providers, sess_options=so)
            self.output_details = [i.name for i in self.model.get_outputs()]
            self.input_details = [i.name for i in self.model.get_inputs()]

            self.is_initiated = True
        except Exception as e:
            raise Exception(f"Cannot load model {model}: {e}")

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)

    def process_mask(self, protos, masks_in, bboxes, shape):
        """
        Process prototype masks with predicted mask coefficients to generate instance segmentation masks.
        
        Args:
            protos (np.ndarray): Prototype masks from model output
            masks_in (np.ndarray): Predicted mask coefficients
            bboxes (np.ndarray): Bounding boxes in xyxy format
            shape (tuple): Original image shape (h, w)
            
        Returns:
            np.ndarray: Binary segmentation masks
        """
        c, mh, mw = protos.shape  # CHW
        
        # Convert to torch tensors for processing
        protos_tensor = torch.from_numpy(protos)
        masks_in_tensor = torch.from_numpy(masks_in)
        
        # Matrix multiplication and reshape
        masks = (masks_in_tensor @ protos_tensor.float().view(c, -1)).view(-1, mh, mw)
        
        # Scale masks to original image size
        ih, iw = shape
        masks = torch.nn.functional.interpolate(masks.unsqueeze(0), (ih, iw), mode='bilinear', align_corners=False)[0]
        
        # Crop masks to bounding boxes
        bboxes_tensor = torch.from_numpy(bboxes).to(torch.int)
        
        final_masks = []
        for i, bbox in enumerate(bboxes_tensor):
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(iw, x2)
            y2 = min(ih, y2)
            
            mask = masks[i]
            mask_crop = mask[y1:y2, x1:x2]
            padded_mask = torch.zeros((ih, iw), device=mask.device, dtype=mask.dtype)
            padded_mask[y1:y2, x1:x2] = mask_crop
            final_masks.append(padded_mask > 0.5)  # Binarize mask
        
        if final_masks:
            return torch.stack(final_masks).cpu().numpy()
        else:
            return np.zeros((0, ih, iw), dtype=bool)

    def _infer(self, image):
        try:
            # Prepare image for inference
            img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            h0, w0 = img.shape[:2]  # Original image shape
            
            # Letterbox and preprocess
            img_lb, ratio, (dw, dh) = self.letterbox(img, self.input_size, auto=False)
            img_lb = img_lb.transpose((2, 0, 1))  # HWC to CHW
            img_lb = np.ascontiguousarray(img_lb)
            img_lb = img_lb.astype(np.float32) / 255.0  # Normalize
            
            # Add batch dimension
            img_lb = np.expand_dims(img_lb, 0)
            
            # Run model inference
            inp = {self.input_details[0]: img_lb}
            outputs = self.model.run(self.output_details, inp)
            
            # Process segmentation outputs
            predictions = outputs[0]  # Detections with shape (1, num_detections, 6+num_classes+mask_dim)
            proto = outputs[1]  # Prototype masks with shape (1, mask_dim, mask_h, mask_w)
            
            # Process predictions
            pred = np.squeeze(predictions, axis=0)  # Remove batch dimension
            proto = np.squeeze(proto, axis=0)  # Remove batch dimension
            
            # Extract boxes, scores, classes, and mask coefficients
            keep_idx = pred[:, 4] > 0.001  # Basic confidence filtering
            if not np.any(keep_idx):
                return [], [], [], []
            
            pred = pred[keep_idx]
            
            # Extract coordinates, confidence scores, class IDs and masks
            boxes = pred[:, :4]  # Boxes in xywh format
            scores = pred[:, 4]
            class_ids = pred[:, 5:5+len(self.labels)].argmax(axis=1)
            mask_coeffs = pred[:, 5+len(self.labels):]  # Mask coefficients
            
            # Convert boxes from center format to corner format
            x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = np.stack([
                x_center - w/2,  # x1
                y_center - h/2,  # y1
                x_center + w/2,  # x2
                y_center + h/2   # y2
            ], axis=1)
            
            # Adjust box coordinates for padding
            boxes[:, [0, 2]] -= dw  # Adjust x coordinates
            boxes[:, [1, 3]] -= dh  # Adjust y coordinates
            
            # Rescale boxes to original image dimensions
            boxes = boxes / ratio
            
            # Process masks
            masks = self.process_mask(proto, mask_coeffs, boxes, (h0, w0))
            
            return boxes, class_ids, scores, masks
            
        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return [], [], [], []

    def to_cvat_mask(self, bbox, mask):
        """
        Converts a binary mask to CVAT's Run-Length Encoded (RLE) format
        
        Args:
            bbox (list): Bounding box in format [x1, y1, x2, y2]
            mask (np.ndarray): Binary mask as numpy array
            
        Returns:
            str: Base64 encoded RLE mask
        """
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        mask_crop = mask[y:y+h, x:x+w]
        
        # RLE encoding
        mask_rle = []
        last_val = 0
        count = 0
        
        for i in range(mask_crop.size):
            # Get value (0 or 1) at position i
            val = mask_crop.flat[i]
            
            # If same as last value, increment count
            if val == last_val:
                count += 1
            else:
                # Store the count of the last value
                mask_rle.append(count)
                # Update last value and reset count
                last_val = val
                count = 1
        
        # Don't forget the last run
        mask_rle.append(count)
        
        # If first value is 1, prepend a 0 count
        if mask_crop.flat[0] == 1:
            mask_rle = [0] + mask_rle
            
        # Convert RLE to bytes
        rle_bytes = bytes(mask_rle)
        
        # Encode as base64
        return base64.b64encode(rle_bytes).decode('utf-8')

    def find_contours(self, mask):
        """
        Find contours in a binary mask
        
        Args:
            mask (np.ndarray): Binary mask
            
        Returns:
            np.ndarray: Contour points
        """
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.array([])
            
        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)
        contour = np.squeeze(contour, axis=1)
        return contour

    def infer(self, image, threshold):
        # Convert PIL image to numpy array
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        h, w = image.shape[:2]
        boxes, class_ids, scores, masks = self._infer(image)
        
        results = []
        
        for box, class_id, score, mask in zip(boxes, class_ids, scores, masks):
            if score < threshold:
                continue
                
            # Convert to integer coordinates
            x1, y1, x2, y2 = map(int, [max(0, box[0]), max(0, box[1]), 
                                      min(w, box[2]), min(h, box[3])])
            
            # Skip too small detections
            if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < 10:
                continue
                
            # Create binary mask and find contours
            binary_mask = mask.astype(np.uint8) * 255
            contour = self.find_contours(binary_mask)
            
            if len(contour) < 3:  # Need at least 3 points for a valid polygon
                continue
                
            # Flip contour to match CVAT format
            contour = np.flip(contour, axis=1)
            
            # Create RLE mask for CVAT
            cvat_mask = self.to_cvat_mask([x1, y1, x2, y2], binary_mask)
            
            results.append({
                "confidence": str(float(score)),
                "label": self.labels.get(int(class_id), "unknown"),
                "points": contour.ravel().tolist(),
                "mask": cvat_mask,
                "type": "mask",
            })
            
        return results
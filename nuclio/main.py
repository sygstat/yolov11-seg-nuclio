import base64
import io
import json
import yaml
from model_handler import ModelHandler
from PIL import Image
import numpy as np

def init_context(context):
    context.logger.info("Init context...  0%")
    
    # Read labels from function.yaml
    with open("/opt/nuclio/function.yaml", 'rb') as function_file:
        functionconfig = yaml.safe_load(function_file)
    
    labels_spec = functionconfig['metadata']['annotations']['spec']
    labels = {item['id']: item['name'] for item in json.loads(labels_spec)}
    
    # Initialize the model handler
    model = ModelHandler(labels)
    context.user_data.model = model
    
    context.logger.info("Init context...100%")

def handler(context, event):
    context.logger.info("Run YOLOv11 Segmentation ONNX model")
    
    # Parse the request data
    data = event.body
    
    try:
        # Decode the base64 image
        buf = io.BytesIO(base64.b64decode(data["image"]))
        
        # Get threshold from request or use default
        threshold = float(data.get("threshold", 0.5))
        
        # Open the image
        image = Image.open(buf)
        
        # Run inference
        results = context.user_data.model.infer(image, threshold)
        
        return context.Response(
            body=json.dumps(results),
            headers={},
            content_type='application/json',
            status_code=200
        )
        
    except Exception as e:
        context.logger.error(f"Error processing request: {str(e)}")
        return context.Response(
            body=json.dumps({"error": str(e)}),
            headers={},
            content_type='application/json',
            status_code=500
        )
import json
import base64
import numpy as np
from PIL import Image
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# 全局变量，避免重复加载模型
model = None
class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '蛙', '马', '船', '卡车']

def load_model_once():
    """只在冷启动时加载模型"""
    global model
    if model is None:
        # 这里的路径要注意，确保你真的上传了 model/cifar_model.h5
        model_path = os.path.join(os.path.dirname(__file__), 'model/cifar_model.h5')
        # 加载模型
        model = load_model(model_path)
    return model

def preprocess_image(image_data):
    # ... (代码同 test1.md)
    image_bytes = base64.b64decode(image_data.split(',')[-1]) # 兼容带前缀或不带的情况
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def main(event, context):
    try:
        model = load_model_once()
        
        image_data = event.get('image', '')
        if not image_data:
            return {'success': False, 'error': 'No image data'}

        # 真实预测逻辑
        processed_image = preprocess_image(image_data)
        predictions = model.predict(processed_image)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        
        return {
            'success': True,
            'predicted_class': predicted_class,
            'class_name': class_names[predicted_class],
            'confidence': confidence,
            'all_predictions': predictions[0].tolist()
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
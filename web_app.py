from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import base64
import torch
from transformers import pipeline
from PIL import Image
import threading
import time

app = Flask(__name__)

class WebDepthCamera:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.depth_estimator = None
        self.model_loaded = False
        self.init_model()
    
    def init_model(self):
        def load_model():
            try:
                device = 0 if torch.cuda.is_available() else -1
                print(f"Loading model on device: {'CUDA' if device == 0 else 'CPU'}")
                self.depth_estimator = pipeline(
                    task="depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=device
                )
                self.model_loaded = True
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Model loading error: {e}")
                self.model_loaded = False
        
        threading.Thread(target=load_model, daemon=True).start()
    
    def start_camera(self):
        self.camera = cv2.VideoCapture(0)
        self.is_running = True
        return self.camera.isOpened()
    
    def stop_camera(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
    
    def get_frame(self):
        if not self.camera or not self.is_running:
            return None, None, "Camera not running"
            
        if not self.model_loaded:
            return None, None, "Model not loaded yet"
            
        ret, frame = self.camera.read()
        if not ret:
            return None, None, "No frame"
            
        # Resize for better quality
        frame = cv2.resize(frame, (640, 480))
        
        try:
            # AI depth estimation
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            depth_result = self.depth_estimator(pil_image)
            depth_map = np.array(depth_result['depth'])
            
            # Normalize and apply heat colormap
            depth_normalized = ((depth_map - depth_map.min()) * (255 / (depth_map.max() - depth_map.min()))).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            
            # Convert to base64
            _, buffer1 = cv2.imencode('.jpg', frame)
            _, buffer2 = cv2.imencode('.jpg', depth_colored)
            
            frame_b64 = base64.b64encode(buffer1).decode('utf-8')
            depth_b64 = base64.b64encode(buffer2).decode('utf-8')
            
            return frame_b64, depth_b64, "OK"
            
        except Exception as e:
            return None, None, f"Depth error: {str(e)}"

camera = WebDepthCamera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    success = camera.start_camera()
    return jsonify({'success': success})

@app.route('/stop')
def stop():
    camera.stop_camera()
    return jsonify({'success': True})

@app.route('/status')
def get_status():
    return jsonify({
        'model_loaded': camera.model_loaded,
        'camera_running': camera.is_running
    })

@app.route('/frame')
def get_frame():
    original, depth, status = camera.get_frame()
    if original and depth:
        return jsonify({
            'original': f'data:image/jpeg;base64,{original}',
            'depth': f'data:image/jpeg;base64,{depth}',
            'status': status
        })
    return jsonify({'error': status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
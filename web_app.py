from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import base64
import torch
from transformers import pipeline
from PIL import Image
import threading
import time
import queue
from collections import deque
import psutil
import gc

app = Flask(__name__)

class OptimizedWebDepthCamera:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.depth_estimator = None
        self.model_loaded = False
        
        # Device management
        self.current_device = "auto"  # auto, cpu, gpu
        self.available_devices = self._detect_available_devices()
        self.device_info = {}
        
        # Performance optimizations
        self.frame_queue = queue.Queue(maxsize=2)  # Limit queue size
        self.depth_queue = queue.Queue(maxsize=2)
        self.latest_frame = None
        self.latest_depth = None
        self.frame_cache = {}
        self.cache_size = 3
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.total_frames = 0
        self.processing_times = deque(maxlen=10)
        
        # Threading
        self.camera_thread = None
        self.depth_thread = None
        self.frame_lock = threading.Lock()
        self.model_lock = threading.Lock()
        
        # Model optimization
        self.model_warmup_done = False
        self.input_size = (384, 384)  # Smaller input for faster processing
        
        self.init_model()
    
    def _detect_available_devices(self):
        """Detect available processing devices"""
        devices = ["cpu"]
        device_info = {
            "cpu": {
                "name": "CPU",
                "available": True,
                "memory": "System RAM"
            }
        }
        
        if torch.cuda.is_available():
            devices.append("gpu")
            device_info["gpu"] = {
                "name": f"GPU - {torch.cuda.get_device_name(0)}",
                "available": True,
                "memory": f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB VRAM"
            }
        else:
            device_info["gpu"] = {
                "name": "GPU - Not Available",
                "available": False,
                "memory": "N/A"
            }
        
        self.device_info = device_info
        return devices
    
    def init_model(self, device_preference=None):
        """Initialize model with specified device preference"""
        def load_model():
            with self.model_lock:
                try:
                    # Determine device to use
                    target_device = device_preference or self.current_device
                    device = self._get_device_for_pipeline(target_device)
                    
                    device_name = self._get_device_display_name(device)
                    print(f"🔥 Loading ARIA Vision model on: {device_name}")
                    
                    # Clear previous model to free memory
                    if hasattr(self, 'depth_estimator') and self.depth_estimator is not None:
                        del self.depth_estimator
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        gc.collect()
                    
                    # Load model with specified device
                    self.depth_estimator = pipeline(
                        task="depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=device,
                        torch_dtype=torch.float16 if device != -1 else torch.float32  # Use FP16 for GPU
                    )
                    
                    # Update current device
                    self.current_device = target_device
                    
                    # Warmup the model with dummy data
                    self.warmup_model()
                    self.model_loaded = True
                    print(f"✅ Model loaded and optimized successfully on {device_name}!")
                    
                except Exception as e:
                    print(f"❌ Model loading error: {e}")
                    self.model_loaded = False
                    # Fallback to CPU if GPU fails
                    if device_preference == "gpu" and torch.cuda.is_available():
                        print("🔄 Falling back to CPU...")
                        self.init_model("cpu")
        
        threading.Thread(target=load_model, daemon=True).start()
    
    def _get_device_for_pipeline(self, device_preference):
        """Convert device preference to pipeline device parameter"""
        if device_preference == "gpu":
            return 0 if torch.cuda.is_available() else -1
        elif device_preference == "cpu":
            return -1
        else:  # auto
            return 0 if torch.cuda.is_available() else -1
    
    def _get_device_display_name(self, device):
        """Get human-readable device name"""
        if device == -1:
            return "CPU"
        elif device == 0:
            return f"GPU ({torch.cuda.get_device_name(0)})"
        else:
            return f"CUDA Device {device}"
    
    def switch_device(self, new_device):
        """Switch processing device (cpu/gpu/auto)"""
        if new_device not in ["cpu", "gpu", "auto"]:
            return {"success": False, "error": "Invalid device. Use 'cpu', 'gpu', or 'auto'"}
        
        if new_device == "gpu" and not torch.cuda.is_available():
            return {"success": False, "error": "GPU not available on this system"}
        
        # Stop processing temporarily
        was_running = self.is_running
        if was_running:
            self.stop_camera()
            time.sleep(0.5)  # Wait for threads to stop
        
        # Reset performance counters
        self.fps_counter.clear()
        self.processing_times.clear()
        self.total_frames = 0
        
        # Reload model with new device
        self.model_loaded = False
        self.model_warmup_done = False
        self.init_model(new_device)
        
        # Wait for model to load
        timeout = 30  # 30 second timeout
        start_time = time.time()
        while not self.model_loaded and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not self.model_loaded:
            return {"success": False, "error": "Model failed to load on new device"}
        
        # Restart camera if it was running
        if was_running:
            time.sleep(0.5)  # Brief pause
            self.start_camera()
        
        return {
            "success": True, 
            "device": self.current_device,
            "device_info": self.device_info[self.current_device if self.current_device != "auto" else ("gpu" if torch.cuda.is_available() else "cpu")]
        }
    
    def warmup_model(self):
        """Warmup model with dummy data for faster inference"""
        if self.depth_estimator:
            try:
                dummy_image = Image.new('RGB', self.input_size, color=(128, 128, 128))
                for _ in range(3):  # Multiple warmup runs
                    _ = self.depth_estimator(dummy_image)
                self.model_warmup_done = True
                print("Model warmup completed")
            except Exception as e:
                print(f"Model warmup failed: {e}")
    
    def start_camera(self):
        """Start camera with optimized settings"""
        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened():
            # Optimize camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
            
            self.is_running = True
            
            # Start dedicated threads
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.depth_thread = threading.Thread(target=self._depth_processing_loop, daemon=True)
            
            self.camera_thread.start()
            self.depth_thread.start()
            
            return True
        return False
    
    def stop_camera(self):
        """Stop camera and cleanup threads"""
        self.is_running = False
        
        if self.camera:
            self.camera.release()
            
        # Clean up queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.depth_queue.empty():
            try:
                self.depth_queue.get_nowait()
            except queue.Empty:
                break
                
        # Clear cache
        self.frame_cache.clear()
        gc.collect()  # Force garbage collection
    
    def _camera_loop(self):
        """Dedicated camera capture thread"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if ret:
                # Update FPS counter
                current_time = time.time()
                if self.last_frame_time:
                    fps = 1.0 / (current_time - self.last_frame_time)
                    self.fps_counter.append(fps)
                self.last_frame_time = current_time
                
                # Put frame in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            time.sleep(0.01)  # Small delay to prevent excessive CPU usage
    
    def _depth_processing_loop(self):
        """Dedicated depth processing thread"""
        while self.is_running:
            if not self.model_loaded or not self.model_warmup_done:
                time.sleep(0.1)
                continue
                
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Resize frame for faster processing
                frame_resized = cv2.resize(frame, self.input_size)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # AI depth estimation
                depth_result = self.depth_estimator(pil_image)
                depth_map = np.array(depth_result['depth'])
                
                # Resize depth map back to original size
                depth_map = cv2.resize(depth_map, (640, 480))
                
                # Normalize and apply heat colormap
                depth_normalized = ((depth_map - depth_map.min()) * (255 / (depth_map.max() - depth_map.min()))).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
                
                # Store processed results
                with self.frame_lock:
                    self.latest_frame = cv2.resize(frame, (640, 480))
                    self.latest_depth = depth_colored
                    self.total_frames += 1
                
                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Depth processing error: {e}")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get latest processed frame (non-blocking)"""
        if not self.camera or not self.is_running:
            return None, None, "Camera not running"
            
        if not self.model_loaded:
            return None, None, "Model not loaded yet"
        
        with self.frame_lock:
            if self.latest_frame is None or self.latest_depth is None:
                return None, None, "No frames processed yet"
            
            # Encode frames to base64
            _, buffer1 = cv2.imencode('.jpg', self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _, buffer2 = cv2.imencode('.jpg', self.latest_depth, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            frame_b64 = base64.b64encode(buffer1).decode('utf-8')
            depth_b64 = base64.b64encode(buffer2).decode('utf-8')
            
            return frame_b64, depth_b64, "OK"
    
    def get_performance_stats(self):
        """Get real-time performance statistics"""
        avg_fps = sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # System stats
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        # GPU stats if available
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        return {
            'fps': round(avg_fps, 1),
            'processing_time_ms': round(avg_processing_time * 1000, 1),
            'total_frames': self.total_frames,
            'cpu_usage': cpu_percent,
            'memory_usage': memory_info.percent,
            'gpu_memory_used': round(gpu_memory_used, 1),
            'gpu_memory_total': round(gpu_memory_total, 1),
            'gpu_memory_percent': round((gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0, 1),
            'model_loaded': self.model_loaded,
            'camera_running': self.is_running,
            'queue_size': self.frame_queue.qsize() if hasattr(self.frame_queue, 'qsize') else 0,
            'current_device': self.current_device,
            'available_devices': self.available_devices,
            'device_info': self.device_info
        }
    
    def get_device_info(self):
        """Get detailed device information"""
        return {
            'current_device': self.current_device,
            'available_devices': self.available_devices,
            'device_info': self.device_info,
            'cuda_available': torch.cuda.is_available(),
            'pytorch_version': torch.__version__
        }

camera = OptimizedWebDepthCamera()

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

@app.route('/performance')
def get_performance():
    """New endpoint for performance monitoring"""
    return jsonify(camera.get_performance_stats())

@app.route('/devices')
def get_devices():
    """Get available devices information"""
    return jsonify(camera.get_device_info())

@app.route('/switch_device', methods=['POST'])
def switch_device():
    """Switch processing device"""
    from flask import request
    data = request.get_json()
    device = data.get('device', 'auto')
    
    result = camera.switch_device(device)
    return jsonify(result)

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
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
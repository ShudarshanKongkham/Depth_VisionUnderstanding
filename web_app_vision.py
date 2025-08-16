from flask import Flask, render_template, Response, jsonify, request
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
import io
import json

app = Flask(__name__)

class ARIAVisionProWithChat:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.depth_estimator = None
        self.vision_model = None
        self.model_loaded = False
        self.vision_model_loaded = False
        
        # Device management
        self.current_device = "auto"  # auto, cpu, gpu
        self.available_devices = self._detect_available_devices()
        self.device_info = {}
        
        # Performance optimizations
        self.frame_queue = queue.Queue(maxsize=2)
        self.depth_queue = queue.Queue(maxsize=2)
        self.latest_frame = None
        self.latest_depth = None
        self.frame_cache = {}
        self.cache_size = 3
        
        # Vision chat capabilities
        self.chat_history = []
        self.last_analysis = {"depth": None, "scene": None, "timestamp": None}
        
        # Performance monitoring
        self.fps_counter = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.total_frames = 0
        self.processing_times = deque(maxlen=10)
        self.vision_processing_times = deque(maxlen=10)
        
        # Threading
        self.camera_thread = None
        self.depth_thread = None
        self.frame_lock = threading.Lock()
        self.model_lock = threading.Lock()
        
        # Model optimization
        self.model_warmup_done = False
        self.vision_warmup_done = False
        self.input_size = (384, 384)  # Smaller input for faster processing
        
        self.init_models()
    
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
    
    def init_models(self, device_preference=None):
        """Initialize both depth and vision models"""
        def load_models():
            with self.model_lock:
                try:
                    # Determine device to use
                    target_device = device_preference or self.current_device
                    device = self._get_device_for_pipeline(target_device)
                    
                    device_name = self._get_device_display_name(device)
                    print(f"🔥 Loading ARIA Vision Pro models on: {device_name}")
                    
                    # Clear previous models to free memory
                    if hasattr(self, 'depth_estimator') and self.depth_estimator is not None:
                        del self.depth_estimator
                    if hasattr(self, 'vision_model') and self.vision_model is not None:
                        del self.vision_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                    
                    # Load depth estimation model
                    print("📊 Loading depth estimation model...")
                    self.depth_estimator = pipeline(
                        task="depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=device,
                        torch_dtype=torch.float16 if device != -1 else torch.float32
                    )
                    
                    # Load vision/chat model (using a smaller, faster model)
                    print("👁️ Loading vision chat model...")
                    self.vision_model = pipeline(
                        task="image-to-text",
                        model="Salesforce/blip-image-captioning-base",
                        device=device,
                        torch_dtype=torch.float16 if device != -1 else torch.float32
                    )
                    
                    # Update current device
                    self.current_device = target_device
                    
                    # Warmup both models
                    self.warmup_models()
                    self.model_loaded = True
                    self.vision_model_loaded = True
                    print(f"✅ All models loaded and optimized successfully on {device_name}!")
                    
                except Exception as e:
                    print(f"❌ Model loading error: {e}")
                    self.model_loaded = False
                    self.vision_model_loaded = False
                    # Fallback to CPU if GPU fails
                    if device_preference == "gpu" and torch.cuda.is_available():
                        print("🔄 Falling back to CPU...")
                        self.init_models("cpu")
        
        threading.Thread(target=load_models, daemon=True).start()
    
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
    
    def warmup_models(self):
        """Warmup both models with dummy data"""
        if self.depth_estimator and self.vision_model:
            try:
                dummy_image = Image.new('RGB', self.input_size, color=(128, 128, 128))
                
                # Warmup depth model
                for _ in range(2):
                    _ = self.depth_estimator(dummy_image)
                self.model_warmup_done = True
                
                # Warmup vision model
                for _ in range(2):
                    _ = self.vision_model(dummy_image)
                self.vision_warmup_done = True
                
                print("🚀 Both models warmed up successfully")
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
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
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
        gc.collect()
    
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
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            time.sleep(0.01)
    
    def _depth_processing_loop(self):
        """Dedicated depth processing thread"""
        while self.is_running:
            if not self.model_loaded or not self.model_warmup_done:
                time.sleep(0.1)
                continue
                
            try:
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
                    
                    # Update last analysis for chat context
                    self.last_analysis["depth"] = depth_map
                    self.last_analysis["timestamp"] = time.time()
                
                # Track processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Depth processing error: {e}")
                time.sleep(0.1)
    
    def analyze_scene(self, instruction="What do you see in this image?"):
        """Analyze current scene with vision model"""
        if not self.vision_model_loaded or not self.vision_warmup_done:
            return {"error": "Vision model not ready"}
        
        with self.frame_lock:
            if self.latest_frame is None:
                return {"error": "No frame available"}
            
            try:
                start_time = time.time()
                
                # Convert frame to PIL Image
                frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Generate scene description
                scene_result = self.vision_model(pil_image)
                scene_description = scene_result[0]['generated_text'] if scene_result else "Unable to analyze scene"
                
                # Enhanced analysis with depth context
                depth_analysis = self._analyze_depth_context()
                
                # Combine vision and depth insights
                combined_analysis = self._combine_vision_depth(scene_description, depth_analysis, instruction)
                
                # Track processing time
                processing_time = time.time() - start_time
                self.vision_processing_times.append(processing_time)
                
                # Update last analysis
                self.last_analysis["scene"] = scene_description
                
                return {
                    "scene_description": scene_description,
                    "depth_analysis": depth_analysis,
                    "combined_analysis": combined_analysis,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }
                
            except Exception as e:
                return {"error": f"Vision analysis failed: {str(e)}"}
    
    def _analyze_depth_context(self):
        """Analyze depth information for context"""
        if self.last_analysis["depth"] is None:
            return "No depth information available"
        
        depth_map = self.last_analysis["depth"]
        
        # Basic depth statistics
        mean_depth = np.mean(depth_map)
        min_depth = np.min(depth_map)
        max_depth = np.max(depth_map)
        
        # Analyze depth distribution
        near_pixels = np.sum(depth_map < mean_depth * 0.7)
        far_pixels = np.sum(depth_map > mean_depth * 1.3)
        total_pixels = depth_map.size
        
        near_percentage = (near_pixels / total_pixels) * 100
        far_percentage = (far_pixels / total_pixels) * 100
        
        # Generate depth insights
        if near_percentage > 40:
            depth_context = "Scene has many close objects"
        elif far_percentage > 40:
            depth_context = "Scene is mostly distant"
        else:
            depth_context = "Scene has mixed depth levels"
        
        return {
            "context": depth_context,
            "near_percentage": round(near_percentage, 1),
            "far_percentage": round(far_percentage, 1),
            "depth_range": round(max_depth - min_depth, 2)
        }
    
    def _combine_vision_depth(self, scene_description, depth_analysis, instruction):
        """Combine vision and depth information for enhanced understanding"""
        
        # Create enhanced response based on instruction and context
        if "distance" in instruction.lower() or "far" in instruction.lower() or "close" in instruction.lower():
            focus = f"Based on depth analysis: {depth_analysis['context']}. "
        elif "objects" in instruction.lower():
            focus = f"I can see: {scene_description}. Depth indicates {depth_analysis['context'].lower()}. "
        else:
            focus = f"{scene_description}. "
        
        # Add depth insights
        if depth_analysis["near_percentage"] > 30:
            focus += f"There are objects close to the camera ({depth_analysis['near_percentage']}% of the scene). "
        if depth_analysis["far_percentage"] > 30:
            focus += f"Much of the scene is in the distance ({depth_analysis['far_percentage']}% far objects). "
        
        return focus.strip()
    
    def process_chat_completion(self, messages, max_tokens=150):
        """Process OpenAI-compatible chat completion request"""
        try:
            # Extract the latest message
            latest_message = messages[-1]
            content = latest_message.get('content', [])
            
            instruction = ""
            image_data = None
            
            # Parse message content
            for item in content:
                if item.get('type') == 'text':
                    instruction = item.get('text', 'What do you see?')
                elif item.get('type') == 'image_url':
                    image_url = item.get('image_url', {}).get('url', '')
                    if image_url.startswith('data:image'):
                        # Extract base64 data
                        image_data = image_url.split(',')[1]
            
            # If no image provided, use current camera frame
            if not image_data:
                analysis = self.analyze_scene(instruction)
            else:
                # Process provided image
                analysis = self._analyze_provided_image(image_data, instruction)
            
            if "error" in analysis:
                response_text = f"Error: {analysis['error']}"
            else:
                response_text = analysis.get('combined_analysis', analysis.get('scene_description', 'Unable to analyze'))
            
            # Add to chat history
            self.chat_history.append({
                "instruction": instruction,
                "response": response_text,
                "timestamp": time.time()
            })
            
            # Keep only last 10 conversations
            if len(self.chat_history) > 10:
                self.chat_history = self.chat_history[-10:]
            
            return {
                "choices": [{
                    "message": {
                        "content": response_text,
                        "role": "assistant"
                    }
                }],
                "usage": {
                    "prompt_tokens": len(instruction.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(instruction.split()) + len(response_text.split())
                }
            }
            
        except Exception as e:
            return {
                "error": {
                    "message": f"Processing failed: {str(e)}",
                    "type": "processing_error"
                }
            }
    
    def _analyze_provided_image(self, image_base64, instruction):
        """Analyze a provided base64 image"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Analyze with vision model
            scene_result = self.vision_model(pil_image)
            scene_description = scene_result[0]['generated_text'] if scene_result else "Unable to analyze image"
            
            # For external images, we can't provide depth analysis
            return {
                "scene_description": scene_description,
                "combined_analysis": f"Image analysis: {scene_description}",
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
    
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
        """Get comprehensive performance statistics"""
        avg_fps = sum(self.fps_counter) / len(self.fps_counter) if self.fps_counter else 0
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        avg_vision_time = sum(self.vision_processing_times) / len(self.vision_processing_times) if self.vision_processing_times else 0
        
        # System stats
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        # GPU stats if available
        gpu_memory_used = 0
        gpu_memory_total = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated(0) / 1024**3
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            'fps': round(avg_fps, 1),
            'processing_time_ms': round(avg_processing_time * 1000, 1),
            'vision_processing_time_ms': round(avg_vision_time * 1000, 1),
            'total_frames': self.total_frames,
            'cpu_usage': cpu_percent,
            'memory_usage': memory_info.percent,
            'gpu_memory_used': round(gpu_memory_used, 1),
            'gpu_memory_total': round(gpu_memory_total, 1),
            'gpu_memory_percent': round((gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0, 1),
            'model_loaded': self.model_loaded,
            'vision_model_loaded': self.vision_model_loaded,
            'camera_running': self.is_running,
            'queue_size': self.frame_queue.qsize() if hasattr(self.frame_queue, 'qsize') else 0,
            'current_device': self.current_device,
            'available_devices': self.available_devices,
            'device_info': self.device_info,
            'chat_history_count': len(self.chat_history)
        }
    
    def switch_device(self, new_device):
        """Switch processing device for both models"""
        if new_device not in ["cpu", "gpu", "auto"]:
            return {"success": False, "error": "Invalid device"}
        
        if new_device == "gpu" and not torch.cuda.is_available():
            return {"success": False, "error": "GPU not available"}
        
        # Stop processing temporarily
        was_running = self.is_running
        if was_running:
            self.stop_camera()
            time.sleep(0.5)
        
        # Reset performance counters
        self.fps_counter.clear()
        self.processing_times.clear()
        self.vision_processing_times.clear()
        self.total_frames = 0
        
        # Reload models with new device
        self.model_loaded = False
        self.vision_model_loaded = False
        self.model_warmup_done = False
        self.vision_warmup_done = False
        self.init_models(new_device)
        
        # Wait for models to load
        timeout = 45  # Longer timeout for both models
        start_time = time.time()
        while (not self.model_loaded or not self.vision_model_loaded) and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if not self.model_loaded or not self.vision_model_loaded:
            return {"success": False, "error": "Models failed to load on new device"}
        
        # Restart camera if it was running
        if was_running:
            time.sleep(0.5)
            self.start_camera()
        
        return {
            "success": True,
            "device": self.current_device,
            "device_info": self.device_info[self.current_device if self.current_device != "auto" else ("gpu" if torch.cuda.is_available() else "cpu")]
        }

# Initialize the enhanced camera system
camera = ARIAVisionProWithChat()

# Routes
@app.route('/')
def index():
    return render_template('vision_chat.html')

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
        'vision_model_loaded': camera.vision_model_loaded,
        'camera_running': camera.is_running
    })

@app.route('/performance')
def get_performance():
    return jsonify(camera.get_performance_stats())

@app.route('/devices')
def get_devices():
    return jsonify({
        'available_devices': camera.available_devices,
        'device_info': camera.device_info,
        'current_device': camera.current_device
    })

@app.route('/switch_device', methods=['POST'])
def switch_device():
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

# @app.route('/analyze_scene', methods=['POST'])
# def analyze_scene():
#     data = request.get_json()
#     instruction = data.get('instruction', 'What do you see?')
#     result = camera.analyze_scene(instruction)
#     return jsonify(result)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    data = request.get_json()
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 150)
    
    result = camera.process_chat_completion(messages, max_tokens)
    return jsonify(result)

@app.route('/chat_history')
def get_chat_history():
    return jsonify({'history': camera.chat_history})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

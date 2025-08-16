import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import queue
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import time
import numpy as np
from collections import deque

class OptimizedCameraGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SmolVLM Camera - Optimized Real-time")
        self.root.geometry("900x700")
        
        # Initialize camera with optimized settings
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # Performance settings
        self.target_inference_size = (224, 224)  # Smaller size for faster inference
        self.analysis_interval = 1.0  # Analyze every N seconds
        self.last_analysis_time = 0
        self.continuous_mode = False
        
        # Frame buffering
        self.frame_buffer = deque(maxlen=3)  # Keep last 3 frames
        self.current_frame = None
        
        # Performance metrics
        self.fps_counter = deque(maxlen=30)
        self.inference_times = deque(maxlen=10)
        self.last_fps_update = time.time()
        self.frame_count = 0
        
        # Threading
        self.analysis_queue = queue.Queue(maxsize=1)  # Only keep latest request
        self.result_queue = queue.Queue()
        self.running = True
        
        self.setup_ui()
        self.load_model_async()
        self.start_camera()
        self.start_analysis_worker()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=5)
        
        # FPS display
        self.fps_label = tk.Label(info_frame, text="FPS: 0", fg='blue', font=('Arial', 10, 'bold'))
        self.fps_label.pack(side=tk.LEFT, padx=5)
        
        # Inference time display
        self.inference_label = tk.Label(info_frame, text="Inference: 0ms", fg='purple', font=('Arial', 10))
        self.inference_label.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = tk.Label(info_frame, text="Loading model...", fg='orange', font=('Arial', 10))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Camera display
        self.camera_label = tk.Label(main_frame, bg='black')
        self.camera_label.pack(pady=5)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Single analyze button
        self.analyze_btn = ttk.Button(control_frame, text="Analyze Once", 
                                     command=self.analyze_single_frame, state='disabled')
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Continuous mode toggle
        self.continuous_btn = ttk.Button(control_frame, text="Start Continuous", 
                                        command=self.toggle_continuous, state='disabled')
        self.continuous_btn.pack(side=tk.LEFT, padx=5)
        
        # Analysis interval slider
        ttk.Label(control_frame, text="Interval (sec):").pack(side=tk.LEFT, padx=(10, 2))
        self.interval_var = tk.DoubleVar(value=1.0)
        self.interval_slider = ttk.Scale(control_frame, from_=0.1, to=3.0, 
                                        variable=self.interval_var, orient=tk.HORIZONTAL,
                                        command=self.update_interval, length=150)
        self.interval_slider.pack(side=tk.LEFT, padx=5)
        self.interval_label = tk.Label(control_frame, text="1.0s")
        self.interval_label.pack(side=tk.LEFT)
        
        # Inference size selector
        ttk.Label(control_frame, text="Quality:").pack(side=tk.LEFT, padx=(10, 2))
        self.quality_var = tk.StringVar(value="fast")
        quality_combo = ttk.Combobox(control_frame, textvariable=self.quality_var, 
                                    values=["fast", "balanced", "quality"], width=10, state='readonly')
        quality_combo.pack(side=tk.LEFT, padx=5)
        quality_combo.bind('<<ComboboxSelected>>', self.update_quality)
        
        # Description frame with larger text area
        desc_frame = ttk.LabelFrame(main_frame, text="Real-time Analysis", padding=10)
        desc_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(desc_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.desc_text = tk.Text(text_frame, height=6, wrap=tk.WORD, font=('Arial', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.desc_text.yview)
        self.desc_text.configure(yscrollcommand=scrollbar.set)
        
        self.desc_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.desc_text.insert('1.0', "Click 'Analyze Once' or 'Start Continuous' to begin AI analysis")
        
    def update_interval(self, value):
        self.analysis_interval = float(value)
        self.interval_label.config(text=f"{self.analysis_interval:.1f}s")
    
    def update_quality(self, event=None):
        quality_settings = {
            "fast": (224, 224),
            "balanced": (336, 336),
            "quality": (448, 448)
        }
        self.target_inference_size = quality_settings[self.quality_var.get()]
    
    def load_model_async(self):
        def load():
            try:
                print("Loading model...")
                self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
                
                # Load model with optimization settings
                self.model = AutoModelForImageTextToText.from_pretrained(
                    "HuggingFaceTB/SmolVLM-500M-Instruct",
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                
                # Set model to evaluation mode
                self.model.eval()
                
                # Compile model for faster inference (PyTorch 2.0+)
                if self.device == "cuda" and hasattr(torch, 'compile'):
                    try:
                        self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
                        print("Model compiled with torch.compile")
                    except Exception as e:
                        print(f"Could not compile model: {e}")
                
                # Warm up the model with a dummy input
                self.warmup_model()
                
                self.model_loaded = True
                self.root.after(0, self.model_ready)
                print("Model loaded successfully")
                
            except Exception as e:
                error_msg = str(e)
                print(f"Model loading error: {error_msg}")
                self.root.after(0, lambda: self.model_error(error_msg))
        
        threading.Thread(target=load, daemon=True).start()
    
    def warmup_model(self):
        """Warm up the model with a dummy input for faster first inference"""
        try:
            dummy_image = Image.new('RGB', self.target_inference_size, color='black')
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe briefly."}
                ]
            }]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[dummy_image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                _ = self.model.generate(
                    **inputs, 
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            print("Model warmed up")
        except Exception as e:
            print(f"Warmup failed (non-critical): {e}")
    
    def model_ready(self):
        self.status_label.config(text="Model Ready", fg='green')
        self.analyze_btn.config(state='normal')
        self.continuous_btn.config(state='normal')
    
    def model_error(self, error):
        self.status_label.config(text=f"Error: {error[:30]}...", fg='red')
    
    def start_camera(self):
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def camera_loop(self):
        """Separate thread for camera capture to ensure smooth video"""
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                # Store original frame for analysis
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = frame_rgb
                
                # Add to buffer for analysis
                self.frame_buffer.append(frame_rgb)
                
                # Update FPS counter
                self.frame_count += 1
                current_time = time.time()
                self.fps_counter.append(current_time)
                
                # Schedule UI update
                self.root.after(0, lambda f=frame_rgb: self.update_display(f))
                
                # Check if we should analyze in continuous mode
                if self.continuous_mode and self.model_loaded:
                    if current_time - self.last_analysis_time >= self.analysis_interval:
                        self.request_analysis()
                        self.last_analysis_time = current_time
            
            time.sleep(0.03)  # ~33 FPS cap
    
    def update_display(self, frame):
        """Update camera display and FPS counter"""
        try:
            # Resize for display
            frame_resized = cv2.resize(frame, (640, 480))
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)
            
            # Update display
            self.camera_label.config(image=photo)
            self.camera_label.image = photo
            
            # Update FPS every second
            current_time = time.time()
            if current_time - self.last_fps_update >= 1.0:
                if len(self.fps_counter) > 1:
                    fps = len(self.fps_counter) / (self.fps_counter[-1] - self.fps_counter[0])
                    self.fps_label.config(text=f"FPS: {fps:.1f}")
                self.last_fps_update = current_time
                
        except Exception as e:
            print(f"Display update error: {e}")
    
    def start_analysis_worker(self):
        """Start background worker for model inference"""
        def worker():
            while self.running:
                try:
                    # Get frame from queue (blocks until available)
                    frame = self.analysis_queue.get(timeout=0.1)
                    
                    if frame is None:
                        continue
                    
                    # Perform inference
                    start_time = time.time()
                    result = self.perform_inference(frame)
                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    # Send result back
                    self.result_queue.put((result, inference_time))
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Analysis worker error: {e}")
                    self.result_queue.put((f"Error: {str(e)[:50]}", 0))
        
        threading.Thread(target=worker, daemon=True).start()
        
        # Start result processor
        self.process_results()
    
    def perform_inference(self, frame):
        """Optimized inference function"""
        try:
            # Convert and resize image
            image = Image.fromarray(frame).resize(self.target_inference_size, Image.LANCZOS)
            
            # Simple prompt for speed
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What do you see? Be brief."}
                ]
            }]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
            
            # Optimized generation settings
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=30,  # Reduced for speed
                    do_sample=False,     # Deterministic for speed
                    num_beams=1,         # No beam search for speed
                    use_cache=True,      # Enable KV cache
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    temperature=0.7,
                    repetition_penalty=1.1
                )
                
                # Decode only the generated part
                generated_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
                response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Inference error: {e}")
            return f"Analysis error: {str(e)[:30]}"
    
    def request_analysis(self):
        """Request analysis of current frame"""
        if self.current_frame is not None and self.model_loaded:
            # Use put_nowait and catch exception if queue is full
            try:
                # Clear queue first to always use latest frame
                while not self.analysis_queue.empty():
                    try:
                        self.analysis_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Add new frame
                self.analysis_queue.put_nowait(self.current_frame.copy())
            except queue.Full:
                pass  # Skip if queue is full
    
    def process_results(self):
        """Process inference results from the queue"""
        try:
            while not self.result_queue.empty():
                result, inference_time = self.result_queue.get_nowait()
                
                # Update inference time display
                self.inference_times.append(inference_time)
                avg_time = np.mean(self.inference_times) if self.inference_times else 0
                self.inference_label.config(text=f"Inference: {avg_time:.0f}ms")
                
                # Update description
                timestamp = time.strftime("%H:%M:%S")
                self.desc_text.delete('1.0', tk.END)
                self.desc_text.insert('1.0', f"[{timestamp}] {result}")
                
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Result processing error: {e}")
        
        # Schedule next check
        if self.running:
            self.root.after(50, self.process_results)
    
    def analyze_single_frame(self):
        """Analyze current frame once"""
        if self.model_loaded and self.current_frame is not None:
            self.request_analysis()
            self.desc_text.delete('1.0', tk.END)
            self.desc_text.insert('1.0', "Analyzing...")
    
    def toggle_continuous(self):
        """Toggle continuous analysis mode"""
        self.continuous_mode = not self.continuous_mode
        
        if self.continuous_mode:
            self.continuous_btn.config(text="Stop Continuous")
            self.analyze_btn.config(state='disabled')
            self.last_analysis_time = time.time()
            self.desc_text.delete('1.0', tk.END)
            self.desc_text.insert('1.0', "Continuous mode active - analyzing in real-time...")
        else:
            self.continuous_btn.config(text="Start Continuous")
            self.analyze_btn.config(state='normal')
            self.desc_text.delete('1.0', tk.END)
            self.desc_text.insert('1.0', "Continuous mode stopped")
    
    def on_closing(self):
        """Clean shutdown"""
        self.running = False
        self.continuous_mode = False
        
        # Give threads time to finish
        time.sleep(0.1)
        
        if self.cap:
            self.cap.release()
        
        # Clear GPU memory if using CUDA
        if self.device == "cuda" and self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        self.root.destroy()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    app = OptimizedCameraGUI()
    app.run()

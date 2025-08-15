import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import queue
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

class CameraGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SmolVLM Camera")
        self.root.geometry("800x600")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        # Threading
        self.analysis_queue = queue.Queue()
        self.running = True
        
        self.setup_ui()
        self.load_model_async()
        self.start_camera()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Camera display
        self.camera_label = tk.Label(main_frame, bg='black')
        self.camera_label.pack(pady=5)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Analyze button
        self.analyze_btn = ttk.Button(control_frame, text="Analyze Frame", 
                                     command=self.analyze_current_frame, state='disabled')
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Loading model...", fg='orange')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Description frame
        desc_frame = ttk.LabelFrame(main_frame, text="Analysis", padding=10)
        desc_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.desc_text = tk.Text(desc_frame, height=4, wrap=tk.WORD)
        self.desc_text.pack(fill=tk.BOTH, expand=True)
        self.desc_text.insert('1.0', "Click 'Analyze Frame' to get AI description")
        
    def load_model_async(self):
        def load():
            try:
                self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
                self.model = AutoModelForImageTextToText.from_pretrained(
                    "HuggingFaceTB/SmolVLM-500M-Instruct",
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
                
                if self.device == "cuda":
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                
                self.model_loaded = True
                self.root.after(0, self.model_ready)
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.model_error(error_msg))
        
        threading.Thread(target=load, daemon=True).start()
    
    def model_ready(self):
        self.status_label.config(text="Ready", fg='green')
        self.analyze_btn.config(state='normal')
    
    def model_error(self, error):
        self.status_label.config(text=f"Error: {error}", fg='red')
    
    def start_camera(self):
        self.update_camera()
    
    def update_camera(self):
        if self.running:
            ret, frame = self.cap.read()
            if ret:
                # Convert and resize for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(image)
                
                # Update display
                self.camera_label.config(image=photo)
                self.camera_label.image = photo
                
                # Store current frame for analysis
                self.current_frame = frame_rgb
            
            self.root.after(30, self.update_camera)  # ~33 FPS
    
    def analyze_current_frame(self):
        print(f"Analyze clicked - Model loaded: {self.model_loaded}, Has frame: {hasattr(self, 'current_frame')}")
        if not self.model_loaded:
            self.desc_text.delete('1.0', tk.END)
            self.desc_text.insert('1.0', "Model not ready yet...")
            return
        if not hasattr(self, 'current_frame'):
            self.desc_text.delete('1.0', tk.END)
            self.desc_text.insert('1.0', "No camera frame available...")
            return
            
        self.analyze_btn.config(state='disabled', text="Analyzing...")
        self.desc_text.delete('1.0', tk.END)
        self.desc_text.insert('1.0', "Analyzing...")
        
        # Start analysis in background
        threading.Thread(target=self.analyze_frame_async, daemon=True).start()
    
    def analyze_frame_async(self):
        print("Starting analysis...")
        try:
            # Resize for faster processing
            image = Image.fromarray(self.current_frame).resize((336, 336), Image.LANCZOS)
            print("Image resized for processing")
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe briefly."}
                ]
            }]
            
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            result = response.split("Assistant:")[-1].strip()
            print(f"Analysis result: {result}")
            self.root.after(0, lambda: self.analysis_complete(result))
            
        except Exception as e:
            print(f"Analysis error: {e}")
            error_msg = str(e)
            self.root.after(0, lambda: self.analysis_complete(f"Error: {error_msg}"))
    
    def analysis_complete(self, result):
        self.desc_text.delete('1.0', tk.END)
        self.desc_text.insert('1.0', result)
        self.analyze_btn.config(state='normal', text="Analyze Frame")
    
    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()
    
    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

if __name__ == "__main__":
    app = CameraGUI()
    app.run()
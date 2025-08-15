import cv2
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import numpy as np

# Global variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None
model_loaded = False

def load_model():
    global processor, model, model_loaded
    try:
        print(f"Loading SmolVLM model on {DEVICE}...")
        processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-500M-Instruct")
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        
        if DEVICE == "cuda":
            model = torch.compile(model, mode="reduce-overhead")
            print("Model compiled for faster inference")
        
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_loaded = False

def analyze_frame(image):
    """Analyze a single frame with SmolVLM"""
    if not model_loaded:
        return "Model not loaded yet"
    
    image = image.resize((336, 336), Image.LANCZOS)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe briefly."}
            ]
        },
    ]
    
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=50,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    return response.split("Assistant:")[-1].strip()

def main():
    load_model()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Press 'q' to quit, 'SPACE' to analyze current frame")
    
    last_description = "Loading model..." if not model_loaded else "Press SPACE to analyze"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display frame with description
        display_frame = frame.copy()
        
        # Add text background
        cv2.rectangle(display_frame, (10, 10), (630, 80), (0, 0, 0), -1)
        
        # Add description text
        words = last_description.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + word) < 70:
                current_line += word + " "
            else:
                lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
            
        for i, line in enumerate(lines[:2]):  # Max 2 lines
            cv2.putText(display_frame, line, (15, 30 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow('SmolVLM Live Camera', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("Analyzing frame...")
            # Convert BGR to RGB and create PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            try:
                last_description = analyze_frame(pil_image)
                print(f"Description: {last_description}")
            except Exception as e:
                last_description = f"Error: {str(e)}"
                print(f"Error analyzing frame: {e}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
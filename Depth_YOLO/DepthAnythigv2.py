import cv2
import torch
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
from PIL import Image

# Initialize the depth estimation pipeline with the smaller, more efficient model
depth_estimator = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf",
    device=0 if torch.cuda.is_available() else -1
)

def process_frame(frame):
    # Convert OpenCV BGR frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    
    # Get depth map
    depth_result = depth_estimator(pil_image)
    # Convert PIL Image to numpy array
    depth_map = np.array(depth_result['depth'])
    # Show min/max before normalization
    # print(f"Raw depth min: {depth_map.min()}, max: {depth_map.max()}")
    # If 'predicted_depth' is available, use it for visualization
    if 'predicted_depth' in depth_result:
        raw_depth = depth_result['predicted_depth'].cpu().numpy()
        print(f"Raw depth min: {raw_depth.min()}, max: {raw_depth.max()}")
        inv_depth = raw_depth.max() - raw_depth
        print(f"Inv Raw depth min: {inv_depth.min()}, max: {inv_depth.max()}")

        # # Show raw and inverted depth side by side
        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.imshow(raw_depth, cmap='jet')
        # plt.title("Raw Predicted Depth")
        # plt.colorbar(fraction=0.046, pad=0.04)
        # plt.subplot(1, 2, 2)
        # plt.imshow(inv_depth, cmap='jet')
        # plt.title("Inverted Raw Predicted Depth")
        # plt.colorbar(fraction=0.046, pad=0.04)
        # plt.tight_layout()
        # plt.show()
    
    # Normalize depth map for visualization (scale to 0-255)
    # depth_map = ((depth_map - depth_map.min()) * (255 / (depth_map.max() - depth_map.min()))).astype(np.uint8)
    # print("Depth shape:  ", depth_map.shape)

    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET) # COLORMAP_INFERNO
    
    # Resize depth map to match input frame size
    depth_colored = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
    
    return depth_colored, raw_depth

# Initialize webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Process frame to get depth map
        depth_visualization, raw_depth = process_frame(frame)

        # Display original and depth side by side
        combined = np.hstack((frame, depth_visualization))
        cv2.imshow('Webcam and Depth', combined)

        # press 'p' to print and show frame and the depth map use matplotlib to show the depth map
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("Frame and Depth Map")
            # put the frame and the depth map side by side
            plt.subplot(1, 2, 1)
            plt.imshow(frame)
            plt.subplot(1, 2, 2)
            plt.imshow(depth_visualization)
            plt.show()

        # Break loop with 'q'
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

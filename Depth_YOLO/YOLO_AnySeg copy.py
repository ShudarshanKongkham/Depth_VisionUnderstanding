"""
This script performs real-time object segmentation and depth estimation using a YOLO model
for object detection and the Depth-Anything-V2 model for depth perception.

It captures video from a webcam, identifies objects in each frame, and overlays
segmentation masks, bounding boxes, and the estimated distance to each object's center.

Key functionalities:
1.  Initialize YOLOv8 and Depth-Anything-V2 models.
2.  Capture video feed from the primary webcam.
3.  For each frame:
    a. Estimate the depth map.
    b. Perform object segmentation using YOLO.
    c. Calculate the distance to the center of each detected object.
    d. Visualize segmentation masks, bounding boxes, and distance labels.
4.  Display the final processed video stream and the depth map side-by-side.
"""

# Import necessary libraries
import cv2
import torch
import numpy as np
import colorsys
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# -----------------------------------------------------------------------------
# 1. MODEL INITIALIZATION AND CONFIGURATION
# -----------------------------------------------------------------------------

print("Initializing models...")

# Check if a CUDA-enabled GPU is available and set the device accordingly
DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")

# Initialize the depth estimation pipeline from Hugging Face Transformers.
# 'Depth-Anything-V2-Small-hf' is a lightweight yet powerful model.
try:
    depth_estimator = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=DEVICE
    )
except Exception as e:
    print(f"Error initializing depth estimator: {e}")
    print("Please ensure you have an internet connection and the necessary libraries installed.")
    exit()

# Load the YOLOv8 segmentation model.
# Ensure the model file 'yolov8s-seg.pt' is in the correct path.
try:
    model = YOLO("yolo11s-seg.pt")
    # Get class names directly from the loaded model
    CLASS_NAMES = model.model.names
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'yolo11s-seg.pt' is in the same directory as the script.")
    exit()

print("Models initialized successfully.")


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def generate_distinct_colors(num_colors):
    """
    Generates a list of n visually distinct BGR colors using HSV color space.

    Args:
        num_colors (int): The number of distinct colors to generate.

    Returns:
        list: A list of n tuples, where each tuple is a BGR color.
    """
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        # Convert HSV to RGB (all values in range [0, 1])
        rgb_float = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        # Convert RGB from [0, 1] to [0, 255] and then to BGR for OpenCV
        bgr_int = (int(rgb_float[2] * 255), int(rgb_float[1] * 255), int(rgb_float[0] * 255))
        colors.append(bgr_int)
    return colors

def process_depth(frame):
    """
    Processes a single frame to estimate its depth map.

    Args:
        frame (np.ndarray): The input BGR frame from OpenCV.

    Returns:
        tuple: A tuple containing:
            - depth_colored (np.ndarray): A colorized depth map for visualization.
            - raw_depth (np.ndarray): The raw, single-channel depth map for distance calculations.
    """
    # Convert frame from OpenCV's BGR format to RGB for the model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    # Get depth estimation from the model
    result = depth_estimator(pil_image)

    # Convert the predicted depth tensor to a NumPy array
    raw_depth = result["predicted_depth"].cpu().numpy()

    # Convert the visualization depth map (PIL Image) to a NumPy array for OpenCV
    depth_for_viz = np.array(result["depth"])

    # Apply a colormap for better visualization
    depth_colored = cv2.applyColorMap(depth_for_viz, cv2.COLORMAP_INFERNO)

    return depth_colored, raw_depth


def get_object_distance(raw_depth, box):
    """
    Calculates the distance to the center of a bounding box using the raw depth map.

    Args:
        raw_depth (np.ndarray): The raw depth map.
        box (list or tuple): The bounding box coordinates [x1, y1, x2, y2].

    Returns:
        float: The relative depth value at the center of the bounding box.
    """
    x1, y1, x2, y2 = map(int, box[:4])
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h, w = raw_depth.shape
    cx = np.clip(cx, 0, w - 1)
    cy = np.clip(cy, 0, h - 1)
    distance = raw_depth[cy, cx]
    return distance

def visualize(frame, results, class_names, class_colors, raw_depth=None):
    """
    Draws segmentation masks, bounding boxes, and labels (including distance) on the frame.

    Args:
        frame (np.ndarray): The original frame.
        results (list): YOLO model detection results.
        class_names (list): List of all possible class names.
        class_colors (list): List of colors for each class.
        raw_depth (np.ndarray, optional): The raw depth map for distance calculation.

    Returns:
        np.ndarray: The frame with visualizations.
    """
    frame_copy = frame.copy()
    for result in results:
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()
            for i in range(len(masks)):
                mask = masks[i]
                box = boxes[i]
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(box[4])
                cls = int(box[5])
                color = class_colors[cls % len(class_colors)]

                # Draw mask
                alpha = 0.4
                resized_mask = cv2.resize(mask, (frame_copy.shape[1], frame_copy.shape[0]))
                colored_mask = np.zeros_like(frame_copy, dtype=np.uint8)
                colored_mask[resized_mask > 0.5] = color
                frame_copy = cv2.addWeighted(frame_copy, 1, colored_mask, alpha, 0)

                # Draw bounding box
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

                # Create label with class name and confidence
                label = f"{class_names[cls]} {conf:.2f}"

                # If raw_depth is provided, calculate distance and add it to the label
                if raw_depth is not None:
                    distance = get_object_distance(raw_depth, box)
                    label += f" | Dist: {distance:.2f}"

                # Draw the final label on the frame
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame_copy

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------

def main():
    """
    Main function to run the webcam capture and processing loop.
    """
    class_colors = generate_distinct_colors(len(CLASS_NAMES))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        # 1. Estimate depth from the current frame
        depth_colored, raw_depth = process_depth(frame)

        # 2. Run YOLO segmentation model
        results = model(frame, stream=True, verbose=False)

        # 3. Visualize results, now passing the raw_depth for distance calculation
        frame_vis = visualize(frame, results, CLASS_NAMES, class_colors, raw_depth)

        # 4. Display the results
        # Resize depth map to match the frame height for clean side-by-side display
        h, w, _ = frame_vis.shape
        depth_colored_resized = cv2.resize(depth_colored, (int(w * (depth_colored.shape[0]/h)), h))
        
        # Combine the visualized frame and the depth map
        combined_display = np.hstack((frame_vis, depth_colored_resized))
        cv2.imshow("YOLO Segmentation with Depth Estimation", combined_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")

if __name__ == "__main__":
    main()

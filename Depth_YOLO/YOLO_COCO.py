import cv2
import numpy as np
from ultralytics import YOLO
import colorsys

# Load COCO class names
with open('Depth_YOLO/coco.names', 'r') as f:
    class_names = f.read().splitlines()

def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

def visualize(frame, results, class_names, class_colors):
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
                label = f"{class_names[cls]} {conf:.2f}"
                cv2.putText(frame_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame_copy

def main():
    model = YOLO("yolo11s-seg.pt")  # Change to your model path if needed
    class_colors = generate_distinct_colors(len(class_names))
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
        results = model(frame, stream=True)
        frame_vis = visualize(frame, results, class_names, class_colors)
        cv2.imshow("YOLO Segmentation & Detection", frame_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
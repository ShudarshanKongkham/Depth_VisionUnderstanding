import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from transformers import pipeline
from PIL import Image

def list_cameras():
    """List all available RealSense cameras"""
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        print(f"  {i}: {dev.get_info(rs.camera_info.name)} - {dev.get_info(rs.camera_info.serial_number)}")
    return len(devices) > 0

def process_with_depthanything(color_frame, depth_estimator):
    """Process frame with DepthAnything model"""
    frame_rgb = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    depth_result = depth_estimator(pil_image)
    
    # Get raw depth and invert values
    if 'predicted_depth' in depth_result:
        raw_depth = depth_result['predicted_depth'].cpu().numpy()
        inv_depth = raw_depth.max() - raw_depth
        # Normalize inverted depth for visualization
        depth_map = ((inv_depth - inv_depth.min()) * (255 / (inv_depth.max() - inv_depth.min()))).astype(np.uint8)
    else:
        depth_map = np.array(depth_result['depth'])
    
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    depth_colored = cv2.resize(depth_colored, (color_frame.shape[1], color_frame.shape[0]))
    
    return depth_colored


def main():
    # List available cameras
    if not list_cameras():
        print("No RealSense cameras found!")
        return
    
    # Initialize DepthAnything model
    print("Loading DepthAnything model...")
    depth_estimator = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=0 if torch.cuda.is_available() else -1
    )
    print("DepthAnything model loaded")
    
    # Initialize pipeline and configuration
    pipeline_rs = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    pipeline_rs.start(config)
    
    try:
        while True:
            # Acquire frames
            frames = pipeline_rs.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # RealSense depth visualization
            rs_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # DepthAnything depth visualization
            ai_depth_colormap = process_with_depthanything(color_image, depth_estimator)
            
            # Get depth value at center
            h, w = depth_image.shape
            center_depth = depth_frame.get_distance(w//2, h//2)
            
            # Add depth text to color image
            cv2.putText(color_image, f"Center Depth: {center_depth:.2f}m", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Create combined views
            rs_combined = np.hstack((color_image, rs_depth_colormap))
            ai_combined = np.hstack((color_image, ai_depth_colormap))
            
            # Display frames
            cv2.imshow('RealSense: Color + Hardware Depth', rs_combined)
            cv2.imshow('DepthAnything: Color + AI Depth', ai_combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        pipeline_rs.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

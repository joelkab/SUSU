#Joel Kabura
# Date: 2023-10-01
# Description:
# This script is a simplified version of MimicPlay for myCobot 280 that doesn't rely on MimicPlay utilities.
# It initializes a RealSense D55 camera, loads high-level and low-level models directly from file paths,
# and provides a simple loop to demonstrate model loading and camera input.
# It generates simulated joint angles for the robot and displays them alongside camera input.
# IF you need the robotic arm to move just confrige the robot make sure its connected any replay the printed joint angles to the robot function.

#!/usr/bin/env python3
"""
Simplified MimicPlay implementation for myCobot 280 that doesn't rely on MimicPlay utilities.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import json  # Added to parse config strings

class SimpleMimicPlayMyCobot:
    def __init__(self, highlevel_model_path, lowlevel_model_path, camera_width=640, camera_height=480, fps=30):
        """
        Initialize simplified MimicPlay for myCobot 280 with RealSense D55 camera.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Store model paths
        self.highlevel_model_path = os.path.abspath(highlevel_model_path)
        self.lowlevel_model_path = os.path.abspath(lowlevel_model_path)
        print(f"High-level model path: {self.highlevel_model_path}")
        print(f"Low-level model path: {self.lowlevel_model_path}")
        
        # Initialize camera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps
        self.initialize_camera()
        
        # Load models directly
        self.load_models_directly()
        
    def initialize_camera(self):
        """Initialize RealSense D55 camera"""
        print("Initializing RealSense D55 camera...")
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Configure streams
            self.config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.fps)
            self.config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.fps)
            
            # Start streaming
            self.profile = self.pipeline.start(self.config)
            
            # Allow camera to warm up
            for i in range(30):
                self.pipeline.wait_for_frames()
            
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Error initializing camera: {e}")
            print("Continuing with simulated camera input")
            self.pipeline = None
            
    def load_models_directly(self):
        """Load model files directly without MimicPlay utilities"""
        print("Loading model files directly...")
        try:
            # Load model files
            self.highlevel_model = torch.load(self.highlevel_model_path, map_location=self.device)
            self.lowlevel_model = torch.load(self.lowlevel_model_path, map_location=self.device)
            
            # Print model information
            print("\nHigh-level model keys:")
            for key in self.highlevel_model.keys():
                print(f"  - {key}")
            
            print("\nLow-level model keys:")
            for key in self.lowlevel_model.keys():
                print(f"  - {key}")
            
            # Extract state dictionaries
            if "state_dict" in self.highlevel_model:
                self.highlevel_state_dict = self.highlevel_model["state_dict"]
                print("\nHigh-level model has state_dict with keys:")
                for key in list(self.highlevel_state_dict.keys())[:10]:  # Show first 10 keys
                    print(f"  - {key}")
                print(f"  ... and {len(self.highlevel_state_dict) - 10} more keys")
            else:
                print("High-level model does not have state_dict")
            
            if "state_dict" in self.lowlevel_model:
                self.lowlevel_state_dict = self.lowlevel_model["state_dict"]
                print("\nLow-level model has state_dict with keys:")
                for key in list(self.lowlevel_state_dict.keys())[:10]:  # Show first 10 keys
                    print(f"  - {key}")
                print(f"  ... and {len(self.lowlevel_state_dict) - 10} more keys")
            else:
                print("Low-level model does not have state_dict")
            
            # Extract config if available
            if "config" in self.highlevel_model:
                print("\nHigh-level model has config")
            
            if "config" in self.lowlevel_model:
                print("Low-level model has config")
            
            print("\nModels loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            sys.exit(1)
            
    def get_camera_observation(self):
        """Get observation from RealSense camera"""
        if self.pipeline is None:
            return self.get_simulated_observation()
        
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                print("Failed to get frames from camera")
                return self.get_simulated_observation()
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (convert to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            
            # Resize images to match the expected input size for the models
            resized_color = cv2.resize(color_image, (84, 84))
            
            return resized_color, depth_colormap
        
        except Exception as e:
            print(f"Error getting camera observation: {e}")
            return self.get_simulated_observation()
        
    def get_simulated_observation(self):
        """Generate simulated observation when camera is not available"""
        print("Generating simulated camera input...")
        
        # Create a simple pattern
        color_image = np.zeros((84, 84, 3), dtype=np.uint8)
        for i in range(84):
            for j in range(84):
                color_image[i, j] = [i * 255 // 84, j * 255 // 84, 128]
        
        depth_colormap = color_image.copy()
        
        return color_image, depth_colormap
        
    def print_model_structure(self):
        """Print the structure of the models"""
        print("\n=== Model Structure Analysis ===")
        
        # Print high-level model structure
        print("\nHigh-level model structure:")
        if "shape_metadata" in self.highlevel_model:
            shape_meta = self.highlevel_model["shape_metadata"]
            print(f"  Shape metadata: {shape_meta}")
        
        # Print low-level model structure
        print("\nLow-level model structure:")
        if "shape_metadata" in self.lowlevel_model:
            shape_meta = self.lowlevel_model["shape_metadata"]
            print(f"  Shape metadata: {shape_meta}")
        
        # Print action dimensions if available
        if "config" in self.lowlevel_model:
            lowlevel_config = self.lowlevel_model["config"]
            # If the config is a string, try to parse it as JSON
            if isinstance(lowlevel_config, str):
                try:
                    lowlevel_config = json.loads(lowlevel_config)
                except Exception as e:
                    print("Error parsing lowlevel model config string:", e)
                    lowlevel_config = {}
            if "algo" in lowlevel_config and "lowlevel" in lowlevel_config["algo"]:
                config_section = lowlevel_config["algo"]["lowlevel"]
                if "action_dim" in config_section:
                    action_dim = config_section["action_dim"]
                    print(f"\nAction dimension: {action_dim}")
                    print(f"This means the model outputs {action_dim} values:")
                    print(f"  - First 6 values: Joint angles for the 6 DOF robot")
                    if action_dim > 6:
                        print(f"  - Value {action_dim}: Gripper command (positive = open, negative = close)")
        
    def run(self, num_steps=10, visualize=True):
        """Run a simple loop to demonstrate model loading and camera input"""
        print(f"\nRunning simple demonstration for {num_steps} steps...")
        
        # Print model structure
        self.print_model_structure()
        
        # Simulate what the output would look like
        print("\n=== Simulated Joint Angle Output ===")
        for step in range(num_steps):
            # Get camera observation
            color_image, depth_image = self.get_camera_observation()
            
            # Display images if requested
            if visualize:
                cv2.imshow('Color', color_image)
                cv2.imshow('Depth', depth_image)
                
                # Exit on ESC
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
            
            # Simulate joint angle output
            print(f"\nStep {step+1}/{num_steps}")
            print("=" * 50)
            print("Joint Angles (degrees):")
            print("-" * 50)
            
            # Generate simulated joint angles based on step
            t = step / 5.0  # Time variable
            joint_angles = [
                np.sin(t) * 160,             # J1: -160 to 160
                np.cos(t) * 160,             # J2: -160 to 160
                np.sin(t * 0.5) * 180,       # J3: -180 to 180
                np.cos(t * 0.5) * 180,       # J4: -180 to 180
                np.sin(t * 0.25) * 180,      # J5: -180 to 180
                np.cos(t * 0.25) * 180,      # J6: -180 to 180
                np.sin(t * 2) * 0.9          # Gripper: -1 to 1
            ]
            
            # Print joint angles
            for i in range(6):
                print(f"Joint {i+1}: {joint_angles[i]:.2f}")
            
            # Print gripper command
            print("-" * 50)
            print(f"Gripper command: {joint_angles[6]:.4f}")
            print(f"Gripper state: {'OPEN' if joint_angles[6] > 0 else 'CLOSED'}")
            print("=" * 50)
            
            # Add a small delay
            time.sleep(0.5)
        
        # Clean up
        if visualize:
            cv2.destroyAllWindows()
        
        # Stop camera
        if self.pipeline is not None:
            self.pipeline.stop()
        
        print("\nDemonstration completed")
        print("To use the actual models for inference, you'll need to properly install MimicPlay")
        print("or modify the script to use the model weights directly")
        
def main():
    parser = argparse.ArgumentParser(description='Simple MimicPlay Model Inspector')
    parser.add_argument('--highlevel', type=str, required=True, help='Path to high-level model checkpoint')
    parser.add_argument('--lowlevel', type=str, required=True, help='Path to low-level model checkpoint')
    parser.add_argument('--steps', type=int, default=10, help='Number of steps to run')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    # Create SimpleMimicPlayMyCobot instance
    mimic = SimpleMimicPlayMyCobot(
        highlevel_model_path=args.highlevel,
        lowlevel_model_path=args.lowlevel
    )
    
    # Run demonstration
    mimic.run(num_steps=args.steps, visualize=not args.no_viz)
    
if __name__ == '__main__':
    main()

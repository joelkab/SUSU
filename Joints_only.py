#!/usr/bin/env python3
"""
MimicPlay implementation for Elephant Robotics myCobot 280 with optional RealSense D55 camera.
This script loads trained high-level and low-level models from MimicPlay and prints
the joint angles without sending commands to the physical robot.

If you don't have a camera for testing, use the --dummy-camera flag to generate dummy observations.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import pyrealsense2 as rs
from pymycobot.mycobot import MyCobot
from collections import OrderedDict
import torch

# Add MimicPlay to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MimicPlay"))

# Import MimicPlay modules
from MimicPlay.utils.file_utils import policy_from_checkpoint
# Removed tensor_utils import as it's not used.
import MimicPlay.utils.obs_utils as ObsUtils

class MimicPlayMyCobot:
    def __init__(self, highlevel_model_path, lowlevel_model_path, camera_width=640, camera_height=480, fps=30, use_dummy_camera=False):
        """
        Initialize MimicPlay for myCobot 280 with optional RealSense D55 camera.
        
        Args:
            highlevel_model_path: Path to the high-level model checkpoint
            lowlevel_model_path: Path to the low-level model checkpoint
            camera_width: Width of camera frames
            camera_height: Height of camera frames
            fps: Camera frames per second
            use_dummy_camera: If True, do not initialize a real camera and use dummy observations
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps
        self.use_dummy_camera = use_dummy_camera
        
        if not self.use_dummy_camera:
            self.initialize_camera()
        else:
            print("Running in dummy camera mode. Skipping camera initialization.")
        
        # Load models
        self.load_models(highlevel_model_path, lowlevel_model_path)
        
        # Initialize observation buffer
        self.obs_buffer = {}
        
    def initialize_camera(self):
        """Initialize RealSense D55 camera"""
        print("Initializing RealSense D55 camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Configure streams
        self.config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.fps)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get camera intrinsics
        self.color_profile = self.profile.get_stream(rs.stream.color)
        self.color_intrinsics = self.color_profile.as_video_stream_profile().get_intrinsics()
        
        # Allow camera to warm up
        for i in range(30):
            self.pipeline.wait_for_frames()
        
        print("Camera initialized successfully")
        
    def load_models(self, highlevel_model_path, lowlevel_model_path):
        """
        Load high-level and low-level models from checkpoints
        
        Args:
            highlevel_model_path: Path to high-level model checkpoint
            lowlevel_model_path: Path to low-level model checkpoint
        """
        print("Loading high-level model...")
        self.highlevel_policy, self.highlevel_ckpt = policy_from_checkpoint(
            ckpt_path=highlevel_model_path,
            device=self.device,
            verbose=True
        )
        
        print("Loading low-level model...")
        self.lowlevel_policy, self.lowlevel_ckpt = policy_from_checkpoint(
            ckpt_path=lowlevel_model_path,
            device=self.device,
            verbose=True
        )
        
        # Extract metadata
        self.highlevel_shape_meta = self.highlevel_ckpt["shape_metadata"]
        self.lowlevel_shape_meta = self.lowlevel_ckpt["shape_metadata"]
        
        print("Models loaded successfully")
        
    def get_camera_observation(self):
        """
        Get observation from RealSense camera or generate a dummy observation
        
        Returns:
            obs: Dictionary containing the observation tensor and dummy robot states
            color_image: Color image (dummy or real)
            depth_image: Depth colormap (dummy or real)
        """
        if self.use_dummy_camera:
            # Create a dummy color image (black image)
            color_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            # Create a dummy depth image (black image)
            depth_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        else:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
            if not depth_frame or not color_frame:
                return None, None, None
                
            # Convert images to numpy arrays
            depth_array = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (convert to 8-bit per pixel first)
            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.03), cv2.COLORMAP_JET)
        
        # Resize images to match the expected input size for the models (84x84)
        resized_color = cv2.resize(color_image, (84, 84))
        
        # Normalize the image (convert to float and scale to [0, 1])
        normalized_color = resized_color.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensor and add batch dimension
        color_tensor = torch.from_numpy(normalized_color).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Create observation dictionary with dummy proprioception data
        obs = {
            "agentview_image": color_tensor,
            "robot0_eef_pos": torch.zeros((1, 3), dtype=torch.float32).to(self.device),
            "robot0_joint_pos": torch.zeros((1, 6), dtype=torch.float32).to(self.device)
        }
            
        return obs, color_image, depth_image
        
    def process_highlevel_output(self, highlevel_output):
        """
        Process output from high-level model to get subgoal for low-level model
        
        Args:
            highlevel_output: Output from high-level model
            
        Returns:
            Processed subgoal for low-level model
        """
        # Extract the subgoal (future trajectory)
        subgoal = highlevel_output.detach().cpu().numpy()
        
        # Create a goal dictionary for the low-level model
        goal_dict = {
            "robot0_eef_pos_future_traj": torch.from_numpy(subgoal).to(self.device)
        }
        
        return goal_dict
        
    def process_lowlevel_output(self, lowlevel_output):
        """
        Process output from low-level model to get joint angles for myCobot
        
        Args:
            lowlevel_output: Output from low-level model
            
        Returns:
            Joint angles for myCobot 280
        """
        # Extract the joint angles
        joint_angles = lowlevel_output.detach().cpu().numpy()
        
        # Scale the joint angles from [-1, 1] to the joint limits of myCobot 280 (in degrees)
        joint_limits = np.array([
            [-160, 160],  # J1
            [-160, 160],  # J2
            [-180, 180],  # J3
            [-180, 180],  # J4
            [-180, 180],  # J5
            [-180, 180],  # J6
        ])
        
        scaled_angles = np.zeros(6)
        for i in range(min(6, joint_angles.shape[0])):
            scaled_angles[i] = np.interp(joint_angles[i], [-1, 1], joint_limits[i])
            
        return scaled_angles
        
    def print_joint_angles(self, joint_angles):
        """
        Print joint angles instead of sending to robot
        
        Args:
            joint_angles: Array of 6 joint angles in degrees
        """
        print("=" * 50)
        print("Joint Angles (degrees):")
        print("-" * 50)
        for i, angle in enumerate(joint_angles):
            print(f"Joint {i+1}: {angle:.2f}")
        print("=" * 50)
            
    def run(self, num_steps=100, visualize=True):
        """
        Run the MimicPlay control loop
        
        Args:
            num_steps: Number of control steps to run
            visualize: Whether to visualize the (dummy or real) camera input
        """
        print(f"Running MimicPlay control loop for {num_steps} steps...")
        
        # Start the policies
        self.highlevel_policy.start_episode()
        self.lowlevel_policy.start_episode()
        
        for step in range(num_steps):
            print(f"Step {step+1}/{num_steps}")
            
            # Get observation from camera or dummy generator
            obs_dict, color_image, depth_image = self.get_camera_observation()
            if obs_dict is None:
                print("Failed to get observation")
                continue
                
            # Get high-level output (subgoal)
            highlevel_output = self.highlevel_policy(ob=obs_dict)
            
            # Process high-level output to get subgoal for low-level model
            goal_dict = self.process_highlevel_output(highlevel_output)
            
            # Get low-level output (joint angles)
            lowlevel_output = self.lowlevel_policy(ob=obs_dict, goal=goal_dict)
            
            # Process low-level output to get joint angles for myCobot
            joint_angles = self.process_lowlevel_output(lowlevel_output)
            
            # Print joint angles instead of sending to robot
            self.print_joint_angles(joint_angles)
            
            # Visualize (if desired)
            if visualize:
                cv2.imshow('Color', color_image)
                cv2.imshow('Depth', depth_image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key to exit
                    break
                    
        # Clean up visualization
        if visualize:
            cv2.destroyAllWindows()
            
        # Stop the camera pipeline if a real camera was used
        if not self.use_dummy_camera:
            self.pipeline.stop()
        
def main():
    parser = argparse.ArgumentParser(description='MimicPlay for myCobot 280 (Print Only)')
    parser.add_argument('--highlevel', type=str, required=True, help='Path to high-level model checkpoint')
    parser.add_argument('--lowlevel', type=str, required=True, help='Path to low-level model checkpoint')
    parser.add_argument('--steps', type=int, default=100, help='Number of control steps to run')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--dummy-camera', action='store_true', help='Use dummy camera input (no physical camera needed)')
    
    args = parser.parse_args()
    
    # Create MimicPlayMyCobot instance with dummy camera flag if specified
    mimic = MimicPlayMyCobot(
        highlevel_model_path=args.highlevel,
        lowlevel_model_path=args.lowlevel,
        use_dummy_camera=args.dummy_camera
    )
    
    # Run control loop
    mimic.run(num_steps=args.steps, visualize=not args.no_viz)
    
if __name__ == '__main__':
    main()



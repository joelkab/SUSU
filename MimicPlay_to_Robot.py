#!/usr/bin/env python3
"""
MimicPlay Implementation with Direct Attribute Extraction and Multi-Camera Support

This script implements the MimicPlay framework for robotic control using a RealSense D55 camera.
It loads high-level and low-level models from checkpoint files, extracts required attributes
directly from demo files, and generates joint angle outputs along with gripper state.

Usage:
    python mimicplay_direct.py --highlevel <path_to_highlevel_model> --lowlevel <path_to_lowlevel_model> --demo <path_to_demo_file> [options]

Options:
    --steps: Number of control steps to run (default: 4)
    --no-viz: Disable visualization
    --debug: Enable debug logging
"""

import os
import sys
import argparse
import numpy as np
import cv2
import torch
import h5py
import pyrealsense2 as rs
import json
from collections import OrderedDict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MimicPlay')

# Add MimicPlay and robomimic to path based on user's directory structure
mimicplay_path = "/home/hqp/MimicPlay"
robomimic_path = "/home/hqp/robomimic"

if os.path.exists(mimicplay_path):
    sys.path.append(mimicplay_path)
    logger.info(f"Added MimicPlay path: {mimicplay_path}")
else:
    logger.warning(f"MimicPlay path not found: {mimicplay_path}")

if os.path.exists(robomimic_path):
    sys.path.append(robomimic_path)
    logger.info(f"Added robomimic path: {robomimic_path}")
else:
    logger.warning(f"robomimic path not found: {robomimic_path}")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Patch missing classes in robomimic
# -----------------------------------------------------------------------------
logger.info("Patching robomimic for compatibility with MimicPlay...")
try:
    import torch.nn as nn
    import robomimic.models.obs_nets

    if not hasattr(robomimic.models.obs_nets, "RNN_MIMO_MLP"):
        class RNN_MIMO_MLP(nn.Module):
            def __init__(self, *args, **kwargs):
                super(RNN_MIMO_MLP, self).__init__()
                logger.info("Created dummy RNN_MIMO_MLP class")

            def forward(self, *args, **kwargs):
                return None
        setattr(robomimic.models.obs_nets, "RNN_MIMO_MLP", RNN_MIMO_MLP)
        logger.info("Added RNN_MIMO_MLP to robomimic.models.obs_nets")

    if not hasattr(robomimic.models.obs_nets, "MIMO_Transformer"):
        class MIMO_Transformer(nn.Module):
            def __init__(self, *args, **kwargs):
                super(MIMO_Transformer, self).__init__()
                logger.info("Created dummy MIMO_Transformer class")

            def forward(self, *args, **kwargs):
                return None
        setattr(robomimic.models.obs_nets, "MIMO_Transformer", MIMO_Transformer)
        logger.info("Added MIMO_Transformer to robomimic.models.obs_nets")

    logger.info("Robomimic patched successfully")
except Exception as e:
    logger.error(f"Failed to patch robomimic: {e}")
    logger.warning("Script may fail when loading MimicPlay models")

# -----------------------------------------------------------------------------
# Helper Functions for Direct Attribute Extraction
# -----------------------------------------------------------------------------
def extract_attributes_from_demo(demo_path, device, num_frames=30):
    if not os.path.exists(demo_path):
        logger.error(f"Demo file not found: {demo_path}")
        return None

    attributes = {
        "goal_image_sequence": None,
        "goal_image_length": num_frames,
        "goal_ee_traj": None,
        "observation_keys":
    }

    try:
        with h5py.File(demo_path, 'r') as f:
            if 'data' not in f:
                logger.error("No 'data' group found in the file.")
                return attributes

            data_group = f['data']
            demo_keys = list(data_group.keys())

            if not demo_keys:
                logger.error("No demo subgroups found in the 'data' group.")
                return attributes

            logger.info(f"Found {len(demo_keys)} demonstrations in {demo_path}: {demo_keys}")

            for demo_key in demo_keys:
                demo_group = data_group[demo_key]
                if 'obs' not in demo_group:
                    logger.warning(f"No observations found in demo {demo_key}")
                    continue

                obs_group = demo_group['obs']
                obs_keys = list(obs_group.keys())
                logger.info(f"Observation keys in demo {demo_key}: {obs_keys}")
                attributes["observation_keys"] = obs_keys

                if 'agentview_image' in obs_group:
                    all_frames = obs_group['agentview_image'][:]
                    total_frames = all_frames.shape[0]
                    logger.info(f"Found {total_frames} frames in demo {demo_key}")

                    if total_frames >= num_frames:
                        frame_indices = range(total_frames - num_frames, total_frames)
                    else:
                        frame_indices = list(range(total_frames))
                        frame_indices.extend([total_frames - 1] * (num_frames - total_frames))

                    goal_images =
                    for idx in frame_indices:
                        image = all_frames[idx]
                        if image.max() > 1.0:
                            image = image.astype(np.float32) / 255.0
                        goal_images.append(image)

                    goal_sequence = np.stack(goal_images)
                    goal_tensor = torch.from_numpy(goal_sequence).permute(0, 3, 1, 2).float().to(device)
                    attributes["goal_image_sequence"] = goal_tensor
                    attributes["goal_image_length"] = len(goal_tensor)
                    logger.info(f"Extracted goal image sequence from demo {demo_key}: {goal_tensor.shape}")

                if 'robot0_eef_pos' in obs_group:
                    eef_pos = obs_group['robot0_eef_pos'][:]
                    if eef_pos.shape[0] > 1:
                        traj_length = min(10, eef_pos.shape[0])
                        goal_ee_traj = eef_pos[-traj_length:]
                        goal_ee_traj_tensor = torch.from_numpy(goal_ee_traj).float().to(device)
                        attributes["goal_ee_traj"] = goal_ee_traj_tensor
                        logger.info(f"Created goal_ee_traj from demo {demo_key}: {goal_ee_traj_tensor.shape}")

                if attributes["goal_image_sequence"] is not None and attributes["goal_ee_traj"] is not None:
                    logger.info(f"Successfully extracted all required attributes from demo {demo_key}")
                    return attributes

    except Exception as e:
        logger.error(f"Error extracting attributes from demo file: {e}")

    return attributes

# -----------------------------------------------------------------------------
# Main MimicPlay Class
# -----------------------------------------------------------------------------
class MimicPlayController:
    def __init__(self, highlevel_model_path, lowlevel_model_path, demo_path=None,
                 camera_width=640, camera_height=480, fps=30):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps

        self.initialize_camera()

        if demo_path and os.path.exists(demo_path):
            logger.info(f"Extracting attributes from demo file: {demo_path}")
            extracted_attributes = extract_attributes_from_demo(demo_path, self.device)
            self.goal_image_sequence = extracted_attributes["goal_image_sequence"]
            self.goal_image_length = extracted_attributes["goal_image_length"]
            self.goal_ee_traj = extracted_attributes["goal_ee_traj"]
            self.observation_keys = extracted_attributes.get("observation_keys",)
            logger.info(f"Observation keys found in demo: {self.observation_keys}")
        else:
            logger.warning("No demo file provided or file not found, using dummy attributes")
            self.goal_image_sequence = torch.zeros((30, 3, 84, 84), device=self.device)
            self.goal_image_length = 30
            self.goal_ee_traj = torch.zeros((10, 3), device=self.device)
            self.observation_keys = ["agentview_image", "robot0_eef_pos", "robot0_joint_pos", "robot0_eye_in_hand_image"]

        self.load_models(highlevel_model_path, lowlevel_model_path)
        self.set_policy_attributes()

    def initialize_camera(self):
        logger.info("Initializing RealSense D55 camera...")
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, self.camera_width, self.camera_height,
                                    rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height,
                                    rs.format.z16, self.fps)

        try:
            self.profile = self.pipeline.start(self.config)
            self.color_profile = self.profile.get_stream(rs.stream.color)
            self.color_intrinsics = self.color_profile.as_video_stream_profile().get_intrinsics()
            for i in range(30):
                self.pipeline.wait_for_frames()
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            raise RuntimeError(f"Failed to initialize RealSense camera: {e}")

    def load_models(self, highlevel_model_path, lowlevel_model_path):
        try:
            from mimicplay.utils.file_utils import policy_from_checkpoint
        except ImportError as e:
            logger.error(f"Error importing from mimicplay: {e}")
            logger.error("This script requires the original MimicPlay repository to be properly installed.")
            raise

        logger.info("\n" + "="*50)
        logger.info("Loading high-level model...")
        logger.info("="*50)

        try:
            self.highlevel_policy, self.highlevel_ckpt = policy_from_checkpoint(
                ckpt_path=highlevel_model_path,
                device=self.device,
                verbose=True
            )
            self.highlevel_model_class = self.highlevel_policy.__class__.__name__
            logger.info(f"High-level model loaded successfully: {self.highlevel_model_class}")
        except Exception as e:
            logger.error(f"Error loading high-level model: {e}")
            raise

        logger.info("\n" + "="*50)
        logger.info("Loading low-level model...")
        logger.info("="*50)

        try:
            try:
                self.lowlevel_policy, self.lowlevel_ckpt = policy_from_checkpoint(
                    ckpt_path=lowlevel_model_path,
                    device=self.device,
                    verbose=True,
                    override_config={"train.hl_checkpoint": highlevel_model_path}
                )
            except TypeError:
                logger.info("override_config not supported, trying without it")
                self.lowlevel_policy, self.lowlevel_ckpt = policy_from_checkpoint(
                    ckpt_path=lowlevel_model_path,
                    device=self.device,
                    verbose=True
                )

            self.lowlevel_model_class = self.lowlevel_policy.__class__.__name__
            logger.info(f"Low-level model loaded successfully: {self.lowlevel_model_class}")
            logger.info("Low-level policy attributes:")
            for attr in dir(self.lowlevel_policy):
                if not attr.startswith('_'):
                    try:
                        value = getattr(self.lowlevel_policy, attr)
                        if not callable(value):
                            logger.info(f"  {attr}: {type(value)}")
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error loading low-level model: {e}")
            raise

        logger.info("\n" + "="*50)
        logger.info("Extracting model metadata...")
        logger.info("="*50)

        self.highlevel_shape_meta = self.highlevel_ckpt.get("shape_metadata", {})
        self.lowlevel_shape_meta = self.lowlevel_ckpt.get("shape_metadata", {})

        logger.info(f"High-level shape metadata: {json.dumps(self.highlevel_shape_meta, indent=2) if isinstance(self.highlevel_shape_meta, dict) else self.highlevel_shape_meta}")
        logger.info(f"Low-level shape metadata: {json.dumps(self.lowlevel_shape_meta, indent=2) if isinstance(self.lowlevel_shape_meta, dict) else self.lowlevel_shape_meta}")

        logger.info("Models loaded successfully")

    def set_policy_attributes(self):
        logger.info("Setting essential attributes on policy objects...")
        if hasattr(self.lowlevel_policy, '__class__'):
            self.lowlevel_policy.goal_image = self.goal_image_sequence
            self.lowlevel_policy.goal_id = 0
            self.lowlevel_policy.goal_image_length = self.goal_image_length
            self.lowlevel_policy.current_id = 0
            self.lowlevel_policy.goal_ee_traj = self.goal_ee_traj

            setattr(self.lowlevel_policy.__class__, 'goal_image', self.goal_image_sequence)
            setattr(self.lowlevel_policy.__class__, 'goal_id', 0)
            setattr(self.lowlevel_policy.__class__, 'goal_image_length', self.goal_image_length)
            setattr(self.lowlevel_policy.__class__, 'current_id', 0)
            setattr(self.lowlevel_policy.__class__, 'goal_ee_traj', self.goal_ee_traj)

            logger.info("Set attributes on low-level policy")

            if hasattr(self.lowlevel_policy, 'policy'):
                self.lowlevel_policy.policy.goal_image = self.goal_image_sequence
                self.lowlevel_policy.policy.goal_id = 0
                self.lowlevel_policy.policy.goal_image_length = self.goal_image_length
                self.lowlevel_policy.policy.current_id = 0
                self.lowlevel_policy.policy.goal_ee_traj = self.goal_ee_traj

                setattr(self.lowlevel_policy.policy.__class__, 'goal_image', self.goal_image_sequence)
                setattr(self.lowlevel_policy.policy.__class__, 'goal_id', 0)
                setattr(self.lowlevel_policy.policy.__class__, 'goal_image_length', self.goal_image_length)
                setattr(self.lowlevel_policy.policy.__class__, 'current_id', 0)
                setattr(self.lowlevel_policy.policy.__class__, 'goal_ee_traj', self.goal_ee_traj)

                logger.info("Set attributes on nested policy")
        logger.info("Policy attributes set successfully")

    def verify_policy_attributes(self):
        if hasattr(self.lowlevel_policy, '__class__'):
            if not hasattr(self.lowlevel_policy, 'goal_image'):
                logger.warning("goal_image missing on low-level policy, fixing...")
                self.lowlevel_policy.goal_image = self.goal_image_sequence
            if not hasattr(self.lowlevel_policy, 'goal_image_length'):
                logger.warning("goal_image_length missing on low-level policy, fixing...")
                self.lowlevel_policy.goal_image_length = self.goal_image_length
            if not hasattr(self.lowlevel_policy, 'goal_id'):
                logger.warning("goal_id missing on low-level policy, fixing...")
                self.lowlevel_policy.goal_id = 0
            if not hasattr(self.lowlevel_policy, 'current_id'):
                logger.warning("current_id missing on low-level policy, fixing...")
                self.lowlevel_policy.current_id = 0
            if not hasattr(self.lowlevel_policy, 'goal_ee_traj'):
                logger.warning("goal_ee_traj missing on low-level policy, fixing...")
                self.lowlevel_policy.goal_ee_traj = self.goal_ee_traj
            if hasattr(self.lowlevel_policy, 'policy'):
                if not hasattr(self.lowlevel_policy.policy, 'goal_image'):
                    logger.warning("goal_image missing on nested policy, fixing...")
                    self.lowlevel_policy.policy.goal_image = self.goal_image_sequence
                if not hasattr(self.lowlevel_policy.policy, 'goal_image_length'):
                    logger.warning("goal_image_length missing on nested policy, fixing...")
                    self.lowlevel_policy.policy.goal_image_length = self.goal_image_length
                if not hasattr(self.lowlevel_policy.policy, 'goal_id'):
                    logger.warning("goal_id missing on nested policy, fixing...")
                    self.lowlevel_policy.policy.goal_id = 0
                if not hasattr(self.lowlevel_policy.policy, 'current_id'):
                    logger.warning("current_id missing on nested policy, fixing...")
                    self.lowlevel_policy.policy.current_id = 0
                if not hasattr(self.lowlevel_policy.policy, 'goal_ee_traj'):
                    logger.warning("goal_ee_traj missing on nested policy, fixing...")
                    self.lowlevel_policy.policy.goal_ee_traj = self.goal_ee_traj

    def get_camera_observation(self):
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                raise RuntimeError("Failed to get valid frames from camera")

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            logger.debug(f"Camera color image shape: {color_image.shape}, dtype: {color_image.dtype}")
            logger.debug(f"Camera depth image shape: {depth_image.shape}, dtype: {depth_image.dtype}")

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            resized_color = cv2.resize(color_image, (84, 84))
            normalized_color = resized_color.astype(np.float32) / 255.0

            logger.debug(f"Normalized color image shape: {normalized_color.shape}, min: {normalized_color.min()}, max: {normalized_color.max()}")

            color_tensor = torch.from_numpy(normalized_color).permute(2, 0, 1).unsqueeze(0).to(self.device)
            logger.debug(f"Color tensor shape: {color_tensor.shape}, device: {color_tensor.device}")

            obs = {
                "agentview_image": color_tensor,
                "robot0_eef_pos": torch.zeros((1, 3), dtype=torch.float32).to(self.device),
                "robot0_joint_pos": torch.zeros((1, 6), dtype=torch.float32).to(self.device),
                "robot0_eye_in_hand_image": color_tensor.clone()
            }

            for key in self.observation_keys:
                if key not in obs:
                    if 'image' in key:
                        obs[key] = color_tensor.clone()
                    elif 'pos' in key or 'position' in key:
                        obs[key] = torch.zeros((1, 3), dtype=torch.float32).to(self.device)
                    elif 'quat' in key or 'orientation' in key:
                        obs[key] = torch.zeros((1, 4), dtype=torch.float32).to(self.device)
                    elif 'joint' in key:
                        obs[key] = torch.zeros((1, 6), dtype=torch.float32).to(self.device)
                    else:
                        obs[key] = torch.zeros((1, 1), dtype=torch.float32).to(self.device)
                    logger.debug(f"Added dummy tensor for observation key: {key}")

            logger.debug(f"Observation dictionary keys: {list(obs.keys())}")
            for key, value in obs.items():
                if isinstance(value, torch.Tensor):
                    logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")

            return obs, color_image, depth_colormap

        except Exception as e:
            logger.error(f"Error getting camera frames: {e}")
            raise RuntimeError(f"Failed to get camera observation: {e}")

    def process_highlevel_output(self, highlevel_output):
        logger.info("Processing high-level output...")
        logger.debug(f"High-level output type: {type(highlevel_output)}")
        if not isinstance(highlevel_output, torch.Tensor):
            highlevel_output = torch.tensor(highlevel_output, device=self.device)

        logger.debug(f"High-level output shape: {highlevel_output.shape}, dtype: {highlevel_output.dtype}, device: {highlevel_output.device}")
        logger.debug(f"High-level output min: {highlevel_output.min().item()}, max: {highlevel_output.max().item()}")

        subgoal = highlevel_output.detach()
        goal_dict = {"robot0_eef_pos_future_traj": subgoal.to(self.device)}

        logger.debug(f"Goal dictionary keys: {list(goal_dict.keys())}")
        for key, value in goal_dict.items():
            if isinstance(value, torch.Tensor):
                logger.debug(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")

        return goal_dict

    def process_lowlevel_output(self, lowlevel_output):
        logger.info("Processing low-level output...")
        logger.debug(f"Low-level output type: {type(lowlevel_output)}")
        if isinstance(lowlevel_output, torch.Tensor):
            logger.debug(f"Low-level output shape: {lowlevel_output.shape}, dtype: {lowlevel_output.dtype}, device: {lowlevel_output.device}")
            logger.debug(f"Low-level output min: {lowlevel_output.min().item()}, max: {lowlevel_output.max().item()}")
            logger.debug(f"Low-level output values: {lowlevel_output.detach().cpu().numpy()}")

        if isinstance(lowlevel_output, torch.Tensor):
            joint_angles = lowlevel_output.detach().cpu().numpy()
        else:
            joint_angles = lowlevel_output

        joint_limits = np.array([
            [-160, 160],
            [-160, 160],
            [-180, 180],
            [-180, 180],
            [-180, 180],
            [-180, 180],
        ])

        scaled_angles = np.zeros(6)
        gripper_state = 0

        if joint_angles.ndim == 1:
            if joint_angles.shape[0] >= 7:
                gripper_cmd = joint_angles[6]
                gripper_state = 1 if gripper_cmd > 0 else 0
                logger.info(f"Gripper command: {gripper_cmd:.4f} -> {'open' if gripper_state else 'closed'}")
            for i in range(min(6, joint_angles.shape[0])):
                scaled_angles[i] = np.interp(joint_angles[i], [-1, 1], joint_limits[i])

        elif joint_angles.ndim == 2:
            if joint_angles.shape[1] >= 7:
                gripper_cmd = joint_angles[0, 6]
                gripper_state = 1 if gripper_cmd > 0 else 0
                logger.info(f"Gripper command: {gripper_cmd:.4f} -> {'open' if gripper_state else 'closed'}")
            for i in range(min(6, joint_angles.shape[1])):
                scaled_angles[i] = np.interp(joint_angles[0, i], [-1, 1], joint_limits[i])
        else:
            logger.error(f"Unexpected joint angles shape: {joint_angles.shape}")
            raise ValueError(f"Unexpected joint angles shape: {joint_angles.shape}")

        logger.info(f"Scaled joint angles: {scaled_angles}")
        return scaled_angles, gripper_state

    def print_joint_angles(self, joint_angles, gripper_state):
        print("=" * 50)
        print("Joint Angles (degrees):")
        print("-" * 50)
        for i, angle in enumerate(joint_angles):
            print(f"Joint {i+1}: {angle:.2f}")
        print(f"Gripper: {'OPEN' if gripper_state else 'CLOSED'}")
        print("=" * 50)

    def run(self, num_steps=4, visualize=True):
        logger.info(f"Running MimicPlay control loop for {num_steps} steps...")

        self.highlevel_policy.start_episode()
        self.lowlevel_policy.start_episode()

        self.set_policy_attributes()

        for step in range(num_steps):
            logger.info(f"\nStep {step+1}/{num_steps}")
            obs_dict, color_image, depth_image = self.get_camera_observation()
            goal_dict = {"agentview_image": obs_dict["agentview_image"].clone()}
            if "robot0_eye_in_hand_image" in obs_dict:
                goal_dict["robot0_eye_in_hand_image"] = obs_dict["robot0_eye_in_hand_image"].clone()
            logger.info("Using current observation as goal")

            logger.info("Calling high-level policy...")
            highlevel_output = self.highlevel_policy(ob=obs_dict, goal=goal_dict)
            subgoal_dict = self.process_highlevel_output(highlevel_output)

            logger.info("Calling low-level policy...")
            self.verify_policy_attributes()
            # Direct call without error handling—any error will stop the program.
            lowlevel_output = self.lowlevel_policy(ob=obs_dict, goal=subgoal_dict)

            joint_angles, gripper_state = self.process_lowlevel_output(lowlevel_output)
            self.print_joint_angles(joint_angles, gripper_state)

            if visualize and color_image is not None and depth_image is not None:
                cv2.imshow('Color', color_image)
                cv2.imshow('Depth', depth_image)
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    logger.info("ESC pressed, stopping...")
                    break

        if visualize:
            cv2.destroyAllWindows()
        self.pipeline.stop()
        logger.info("Control loop finished")


def main():
    parser = argparse.ArgumentParser(description='MimicPlay with RealSense D55 camera')
    parser.add_argument('--highlevel', type=str, required=True, help='Path to high-level model checkpoint')
    parser.add_argument('--lowlevel', type=str, required=True, help='Path to low-level model checkpoint')
    parser.add_argument('--demo', type=str, help='Path to demonstration HDF5 file')
    parser.add_argument('--steps', type=int, default=4, help='Number of control steps to run')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    if not args.highlevel or not args.lowlevel:
        logger.error("Both --highlevel and --lowlevel arguments are required")
        parser.print_help()
        return

    try:
        controller = MimicPlayController(
            highlevel_model_path=args.highlevel,
            lowlevel_model_path=args.lowlevel,
            demo_path=args.demo
        )

        controller.run(
            num_steps=args.steps,
            visualize=not args.no_viz
        )

    except Exception as e:
        logger.error(f"Error running MimicPlay controller: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
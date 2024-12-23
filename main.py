import argparse
import cv2
import numpy as np
from tqdm import tqdm
import pickle
import os

from video_utils import extract_frames, save_video, visualize_flow
from optical_flow import estimate_optical_flow_raft, estimate_optical_flow_farneback, estimate_optical_flow_deepflow
from motion_compensation import compensate_motion
import motion_check

def parse_arguments():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(description="Video stabilization using optical flow")
    parser.add_argument("--video", help="Path to the input video", required=True, type=str)
    parser.add_argument("--output", help="Name of the output video (default: output.avi)", type=str, default="output.avi")
    parser.add_argument("--method", help="Optical flow method to use (default: raft)", 
                        type=str, choices=["raft", "farneback", "deepflow"], default="raft")
    parser.add_argument("--visualize_flow", help="Visualize optical flow", action="store_true")
    parser.add_argument("--analyze_stabilization", help="Analyze video stabilization", action="store_true")
    parser.add_argument("--save_flow", type=str, help="File to save the calculated flow.")
    parser.add_argument("--load_flow", type=str, help="File to load the precomputed flow.")
    return parser.parse_args()


def save_flow(flow, flow_file):
    """Save the flow to a file."""
    with open(flow_file, 'wb') as f:
        pickle.dump(flow, f)
    print(f"Flow saved to {flow_file}")


def load_flow(flow_file):
    """Load the flow from a file."""
    if not os.path.exists(flow_file):
        raise FileNotFoundError(f"Flow file not found: {flow_file}")
    with open(flow_file, 'rb') as f:
        flow = pickle.load(f)
    print(f"Flow loaded from {flow_file}")
    return flow

def main(args):

    #* EXTRACTION
    frames, fps, dimensions = extract_frames(args.video)

    #* FLOW ESTIMATION
    optical_flow_methods = {
        "raft": estimate_optical_flow_raft,
        "farneback": estimate_optical_flow_farneback,
        "deepflow": estimate_optical_flow_deepflow,
    }

    flow_method = optical_flow_methods[args.method]

    if args.load_flow:
        # Load precomputed flow
        flows = load_flow(args.load_flow)
    else:
        # Calculate flow and optionally save it
        flows = flow_method(frames)
        if args.save_flow:
            save_flow(flows, args.save_flow)

    #* VISUALIZE OPTICAL FLOW (if the '--visualize_flow' flag is set)
    if args.visualize_flow:
        visualized_flow_frames = visualize_flow(flows)
        save_video(visualized_flow_frames, f"output/visualized_flow_{args.method}.avi", fps, dimensions)

    if args.analyze_stabilization:
        analyze_stabilization_flag = True
    else:
        analyze_stabilization_flag = False

    #* MOTION COMPENSATION
    stabilized_frames = compensate_motion(frames, flows, analyze_stabilization_flag)
    save_video(stabilized_frames, f"output/stabilized_video_{args.method}.avi", fps, dimensions)

    # stabilized_flows = flow_method(stabilized_frames)
    # motion_check.plot_motion(flows, stabilized_flows)

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
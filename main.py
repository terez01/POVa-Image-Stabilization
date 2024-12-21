import argparse
import cv2
import numpy as np
from tqdm import tqdm

from video_utils import extract_frames, save_video, visualize_flow
from optical_flow import estimate_optical_flow_raft, estimate_optical_flow_farneback, estimate_optical_flow_deepflow
from motion_compensation import compensate_motion

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
    return parser.parse_args()


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

    flows = flow_method(frames)

    #* VISUALIZE OPTICAL FLOW (if the '--visualize_flow' flag is set)
    if args.visualize_flow:
        visualized_flow_frames = visualize_flow(flows)
        save_video(visualized_flow_frames, f"output/visualized_flow_{args.method}.avi", fps, dimensions)

    #* MOTION COMPENSATION, flow_shape:(H, W, 2)
    stabilized_frames = compensate_motion(frames, flows)
    save_video(stabilized_frames, f"output/stabilized_video_{args.method}.avi", fps, dimensions)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
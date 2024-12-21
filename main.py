import argparse
import cv2
import numpy as np
from tqdm import tqdm

from video_utils import extract_frames, save_video, visualize_flow
from optical_flow import estimate_optical_flow_raft, estimate_optical_flow_farneback, estimate_optical_flow_deepflow


def parse_arguments():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser(description="Video stabilization using optical flow")
    parser.add_argument("--video", help="Path to the input video", required=True, type=str)
    parser.add_argument("--output", help="Name of the output video (default: output.avi)", type=str, default="output.avi")
    parser.add_argument("--method", help="Optical flow method to use (default: raft)", 
                        type=str, choices=["raft", "farneback", "deepflow"], default="raft")
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

    #* VISUALIZE OPTICAL FLOW
    visualized_flow_frames = visualize_flow(flows)

    # the output dimensions of optical flow estimated with raft are different than the initial one
    if args.method == "raft":
        save_video(visualized_flow_frames, f"output/visualized_flow_{args.method}.avi", fps, (960, 520))
    else:
        save_video(visualized_flow_frames, f"output/visualized_flow_{args.method}.avi", fps, dimensions)

    #* VYPOCET POSUNOV MEDZI SNIMKAMI


    #* FIX BORDER ARTEFACTS




if __name__ == '__main__':
    args = parse_arguments()
    main(args)
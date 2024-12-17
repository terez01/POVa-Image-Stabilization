import argparse
import cv2
import numpy as np
from tqdm import tqdm

from utils.video_utils import play_video_from_path, play_video_from_frames, save_video_from_frames
from preprocessing import preprocess_video
from optical_flow import estimate_flow_sea_raft, estimate_flow_farneback, estimate_flow_lucas_kanade, estimate_flow_deepflow, visualize_flow
#todo: motion compensation
#todo: deblur
#todo: image warp

# from raft import RAFT
# from utils.utils import load_ckpt

# example: python3 main.py --video /home/tomas/1mit/pova/image_stabilization/POVa-Image-Stabilization/input/53_short.mp4 --optical_flow_method farneback --model searaft_models/Tartan-C-T-TSKH-spring540x960-M.pth
def parse_arguments():
    """
    Parses arguments.
    """
    parser = argparse.ArgumentParser(description="Video stabilization using SEA-RAFT or Farneback optical flow")
    parser.add_argument("--video", help="Path to the input video", required=True, type=str)
    parser.add_argument("--model", help="Path to the SEA-RAFT model checkpoint", type=str, default=None)
    parser.add_argument("--device", help="Device to run the model on (default: cpu)", type=str, default="cpu")
    parser.add_argument("--output", help="Name of the output video (default: output.avi)", type=str, default="output.avi")
    parser.add_argument("--optical_flow_method", help="Optical flow method to use (default: sea-raft)", 
                        type=str, choices=["sea-raft", "farneback", "lucas-kanade", "deepflow"], default="sea-raft")
    return parser.parse_args()


def main(args):    

    #* PREPROCESS
    frames, original_frames, fps, dimensions = preprocess_video(args.video, scale_factor=1.0)

    #* ESTIMATE OPTICAL FLOW
    optical_flow_methods = {
        "sea-raft": estimate_flow_sea_raft,
        "farneback": estimate_flow_farneback,
        "lucas-kanade": estimate_flow_lucas_kanade,
        "deepflow": estimate_flow_deepflow,
    }

    flow_method = optical_flow_methods[args.optical_flow_method]
    flows = flow_method(frames)

    #todo: if flow method == sea-raft:
    # load model
        # model = RAFT(args)
        # load_ckpt(model, args.path)
        # model.eval()
        # flows = flow_method(model,frames)

    #* VISUALIZE OPTICAL FLOW
    #todo change to be optional via argument
    visualized_flow_frames = visualize_flow(flows)
    save_video_from_frames(visualized_flow_frames, "output/visualized_flow.avi", fps, dimensions)
    play_video_from_path("output/visualized_flow.avi")

    #todo: MOTION COMPENSATION
    # stabilized_frames = compensate_motion(original_frames, smoothed_flows)

    #todo: DEBLUR
    # apply wiener filter to the frames

    #todo: IMAGE WARPING
    # warped_frames = warp_frames(stabilized_frames)

    #* SHOW AND SAVE THE RESULT
    # play_video_from_frames(warped_frames, fps)
    # save_video_from_frames(warped_frames, args.output, fps, dimensions)
        
if __name__ == '__main__':
    args = parse_arguments()
    main(args)
import cv2
import numpy as np
from tqdm import tqdm

from utils.flow_viz import flow_to_image

# TODO
def estimate_flow_sea_raft(model, frames):
    """
    Placeholder for optical flow estimation using SEA RAFT model.

    Args:
        model: Pretrained SEA RAFT model for optical flow.
        frames (list): List of video frames (numpy arrays).

    Returns:
        int: Currently returns 0 as this is a placeholder.
    """
    # Example usage:
    # flow = model(frame1, frame2)  # Optical flow estimation
    # return flow
    return 0


def estimate_flow_farneback(frames):
    """
    Estimates optical flow using the Farneback method.

    Args:
        frames (list): List of video frames (numpy arrays).

    Returns:
        list: Optical flow between consecutive frames, where each flow 
              is a 2D array of shape (height, width, 2). 
              - The first channel ([..., 0]) contains horizontal displacement.
              - The second channel ([..., 1]) contains vertical displacement.
    """
    flows = []

    for i in tqdm(range(len(frames) - 1), desc="Estimating Optical Flow (Farneback)"):
        flow = cv2.calcOpticalFlowFarneback(
            frames[i], frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    
    return flows

#! DOESNT WORK YET - optional task
def estimate_flow_lucas_kanade(frames):
    """
    Estimates optical flow using the Lucas-Kanade method.

    Args:
        frames (list): List of video frames (numpy arrays).

    Returns:
        list: Optical flow between consecutive frames, represented as a list 
              of 3D arrays of shape (height, width, 2), with horizontal and 
              vertical displacements.
    """
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    flows = []

    for i in tqdm(range(len(frames) - 1), desc="Estimating Optical Flow (Lucas-Kanade)"):
        p0 = cv2.goodFeaturesToTrack(frames[i], mask=None, maxCorners=100, qualityLevel=0.3, minDistance=7)
        p1, st, err = cv2.calcOpticalFlowPyrLK(frames[i], frames[i + 1], p0, None, **lk_params)

        # Initialize flow as a 3D array with shape (height, width, 2)
        flow = np.zeros((frames[i].shape[0], frames[i].shape[1], 2), dtype=np.float32)

        for j, good_new in enumerate(p1):
            a, b = good_new.ravel()
            flow[int(p0[j][0][1]), int(p0[j][0][0]), 0] = a - p0[j][0][0]  # X displacement
            flow[int(p0[j][0][1]), int(p0[j][0][0]), 1] = b - p0[j][0][1]  # Y displacement

        flows.append(flow)
    
    return flows


def estimate_flow_deepflow(frames):
    """
    Estimates optical flow using the DeepFlow method.

    Args:
        frames (list): List of video frames (numpy arrays).

    Returns:
        list: Optical flow between consecutive frames, where each flow 
              is a 2D array of shape (height, width, 2).
    """
    dis = cv2.optflow.createOptFlow_DeepFlow()
    flows = []

    for i in tqdm(range(len(frames) - 1), desc="Estimating Optical Flow (DeepFlow)"):
        flow = dis.calc(frames[i], frames[i + 1], None)
        flows.append(flow)

    return flows


def visualize_flow(flows):
    """
    Converts optical flow to RGB images for visualization.

    Args:
        flows (list): List of optical flow fields (numpy arrays), where 
                      each flow has shape (height, width, 2).

    Returns:
        list: List of visualized frames (numpy arrays in RGB format).
    """
    visualized_frames = []

    for flow in tqdm(flows, desc="Visualizing Optical Flow"):
        vis_frame = flow_to_image(flow)
        visualized_frames.append(vis_frame)

    return visualized_frames
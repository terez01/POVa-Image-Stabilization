import cv2
import numpy as np
from tqdm import tqdm

from conversion_utils import frames_to_tensor

import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def estimate_optical_flow_raft(frames, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    @brief Estimates optical flow using the RAFT method.

    @param frames List of video frames.
    @return Estimated optical flow between consecutive frames.

    source: https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html
    """
    # Initialize RAFT model
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model.eval()
    
    frames_tensor = frames_to_tensor(frames)

    # Preprocessing necessary for RAFT
    def preprocess(batch):
        transforms = T.Compose([
            T.ConvertImageDtype(torch.float32),
            # T.Resize(size=(520, 960)),        # no need to resize, resolution constraints were checked in extraction
            T.Normalize(mean=0.5, std=0.5),     # map [0, 1] into [-1, 1]
        ])
        batch = transforms(batch)
        return batch

    frames_tensor = preprocess(frames_tensor).to(device)

    flows = []

    for i in tqdm(range(len(frames_tensor) - 1), desc="Estimating Optical Flow (RAFT)", dynamic_ncols=True):
        img1 = frames_tensor[i:i+1]  # Shape (1, C, H, W)
        img2 = frames_tensor[i+1:i+2]  # Shape (1, C, H, W)
        
        with torch.no_grad():
            # predicting optical flow using RAFT
            flow = model(img1, img2)[-1][0].cpu().numpy()

            # ensure it has the same shape as other methods (H, W, 2)
            flow = np.transpose(flow, (1, 2, 0))

        flows.append(flow)

    return flows


def estimate_optical_flow_farneback(frames):
    """
    @brief Estimates optical flow using the Farneback method.

    @param frames List of video frames.

    @return Estimated optical flow between consecutive frames.
    """
    # Farneback needs grayscale frames
    grayscale_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    
    flows = []

    for i in tqdm(range(len(grayscale_frames) - 1), desc="Estimating Optical Flow (Farneback)", dynamic_ncols=True):
        flow = cv2.calcOpticalFlowFarneback(
            grayscale_frames[i], grayscale_frames[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)
    
    return flows


def estimate_optical_flow_deepflow(frames):
    """
    @brief Estimates optical flow using the DeepFlow method.

    @param frames List of video frames.

    @return Estimated optical flow between consecutive frames.
    """
    # Deepflow needs grayscale frames
    grayscale_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

    flows = []
    deepflow = cv2.optflow.createOptFlow_DeepFlow()

    for i in tqdm(range(len(grayscale_frames) - 1), desc="Estimating Optical Flow (DeepFlow)", dynamic_ncols=True):
        flow = deepflow.calc(grayscale_frames[i], grayscale_frames[i + 1], None)
        flows.append(flow)

    return flows

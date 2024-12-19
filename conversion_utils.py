import torch


def frames_to_tensor(frames):
    """
    @brief Converts a list of frames to a tensor of shape (N, C, H, W).

    @param frames List of video frames, each frame being a numpy array of shape (H, W, C).

    @return A tensor of shape (N, C, H, W), where N is the number of frames,
            C is the number of channels (RGB), H is the height, and W is the width.
    """
    # Convert each frame to a tensor and permute the dimensions (H, W, C) -> (C, H, W)
    frames_tensor = [torch.tensor(frame).permute(2, 0, 1) for frame in frames]
    
    # Stack all frames into a single tensor of shape (N, C, H, W)
    frames_tensor = torch.stack(frames_tensor)
    
    return frames_tensor


def flows_to_tensors(flows):
    """
    @brief Converts a list of optical flow arrays (NumPy) to PyTorch tensors.

    @param flows List of optical flow arrays (NumPy format).

    @return List of optical flow tensors (PyTorch format).
    """
    flow_tensors = []

    for flow in flows:
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # Shape (2, H, W)
        flow_tensors.append(flow_tensor)
    
    return flow_tensors

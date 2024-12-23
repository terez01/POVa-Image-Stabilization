import cv2
import numpy as np
from tqdm import tqdm

def compensate_motion(frames, flows, smoothness_weight):
    """
    @brief Stabilizes video frames using optical flow data.

    @details This function performs video stabilization through these steps:
    1. Pads frames to accommodate motion compensation
    2. Tracks cumulative camera motion using optical flow
    3. Optimizes the motion path to create smoother camera movement
    4. Applies transforms to stabilize frames while maintaining frame content

    @param frames List of video frames.
    @param flows List of optical flow arrays.

    @return List of stabilized video frames.
    """

    h, w = frames[0].shape[:2]

    # 50% padding for accomodating motion
    pad = int(max(h, w) * 0.5)  

    padded_frames = [cv2.copyMakeBorder(
        frame, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE
    ) for frame in frames]

    positions_x = [0]
    positions_y = [0]
    
    # cumulative camera motion
    for flow in flows:
        dx = np.mean(flow[:, :, 0])
        dy = np.mean(flow[:, :, 1])
        positions_x.append(positions_x[-1] + dx)
        positions_y.append(positions_y[-1] + dy)
    
    positions_x = np.array(positions_x)
    positions_y = np.array(positions_y)
    
    from scipy.optimize import minimize
    
    def objective(smooth_path, original_path, smoothness_weight):
        """
        @brief Objective function for path optimization
        
        @param smooth_path Candidate smoothed motion path
        @param original_path Original motion path

        @return Combined error score from path difference and acceleration
        """
        
        path_error = np.sum((smooth_path - original_path) ** 2)
        
        acceleration = smooth_path[2:] - 2 * smooth_path[1:-1] + smooth_path[:-2]
        smoothness_error = np.sum(acceleration ** 2)
        
        return path_error + smoothness_weight * smoothness_error
    
    # optimize motion path
    smooth_x = minimize(lambda x: objective(x, positions_x, smoothness_weight), positions_x.copy(), method='BFGS').x
    smooth_y = minimize(lambda x: objective(x, positions_y, smoothness_weight), positions_y.copy(), method='BFGS').x
    
    # adjustment for initial padding that had some trouble fitting to frame
    smooth_x += pad * -0.1
    
    stabilized_frames = []
    for i in tqdm(range(len(flows)), desc="Stabilizing Video"):
        frame = padded_frames[i + 1]
        
        # get transformation matrix
        transform = np.array([
            [1, 0, -(positions_x[i+1] - smooth_x[i+1])],
            [0, 1, -(positions_y[i+1] - smooth_y[i+1])]
        ], dtype=np.float32)
        
        h_pad, w_pad = frame.shape[:2]
        stabilized = cv2.warpAffine(
            frame, transform, (w_pad, h_pad),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_TRANSPARENT
        )
        
        # cropping to fit frame
        y_start = (h_pad - h) // 2
        x_start = (w_pad - w) // 2
        stabilized = stabilized[y_start:y_start+h, x_start:x_start+w]
        
        stabilized_frames.append(stabilized)

    return stabilized_frames
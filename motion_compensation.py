import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_motion_analysis(frames, flows, stabilized_frames):
    """Analyze and plot motion before and after stabilization"""
    original_x = [0]
    original_y = [0]
    stabilized_x = [0]
    stabilized_y = [0]
    
    # Calculate motion for original and stabilized sequences
    for i in range(len(flows)):
        # Original motion
        dx_orig = np.mean(flows[i][:, :, 0])
        dy_orig = np.mean(flows[i][:, :, 1])
        original_x.append(original_x[-1] + dx_orig)
        original_y.append(original_y[-1] + dy_orig)
        
        # Calculate motion between stabilized frames
        if i < len(stabilized_frames) - 1:
            flow_stab = cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(stabilized_frames[i], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(stabilized_frames[i+1], cv2.COLOR_BGR2GRAY),
                None, 0.3, 3, 30, 3, 5, 1.2, 0
            )
            dx_stab = np.mean(flow_stab[:, :, 0])
            dy_stab = np.mean(flow_stab[:, :, 1])
            stabilized_x.append(stabilized_x[-1] + dx_stab)
            stabilized_y.append(stabilized_y[-1] + dy_stab)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot trajectories
    plt.subplot(211)
    plt.plot(original_x, label='Original X')
    plt.plot(stabilized_x, label='Stabilized X')
    plt.title('Horizontal Motion Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(212)
    plt.plot(original_y, label='Original Y')
    plt.plot(stabilized_y, label='Stabilized Y')
    plt.title('Vertical Motion Over Time')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Calculate metrics
    orig_acceleration_x = np.diff(np.diff(original_x))
    orig_acceleration_y = np.diff(np.diff(original_y))
    stab_acceleration_x = np.diff(np.diff(stabilized_x))
    stab_acceleration_y = np.diff(np.diff(stabilized_y))
    
    stability_improvement_x = (np.var(orig_acceleration_x) - np.var(stab_acceleration_x)) / np.var(orig_acceleration_x) * 100
    stability_improvement_y = (np.var(orig_acceleration_y) - np.var(stab_acceleration_y)) / np.var(orig_acceleration_y) * 100
    
    print(f"Stability Improvement:")
    print(f"Horizontal: {stability_improvement_x:.1f}%")
    print(f"Vertical: {stability_improvement_y:.1f}%")
    
    return plt



def get_accumulated_transform(flows):
    """Calculate cumulative transforms with centering"""
    transforms = []
    
    # Accumulate total motion
    total_dx = 0
    total_dy = 0
    
    # Store all positions for computing bounds
    all_positions_x = [0]
    all_positions_y = [0]
    
    # First pass: calculate raw transforms
    for flow in flows:
        # Get more robust motion estimate using multiple methods
        dx = np.mean(flow[:, :, 0])  # Average motion
        dy = np.mean(flow[:, :, 1])
        
        # Update cumulative position
        total_dx += dx
        total_dy += dy
        
        # Store positions
        all_positions_x.append(total_dx)
        all_positions_y.append(total_dy)
    
    # Calculate bounds to keep frame in view
    min_x = min(all_positions_x)
    max_x = max(all_positions_x)
    min_y = min(all_positions_y)
    max_y = max(all_positions_y)
    
    # Center the motion range
    center_offset_x = -(max_x + min_x) / 2
    center_offset_y = -(max_y + min_y) / 2
    
    # Second pass: create centered transforms
    total_dx = 0
    total_dy = 0
    
    for flow in flows:
        dx = np.mean(flow[:, :, 0])
        dy = np.mean(flow[:, :, 1])
        
        total_dx += dx
        total_dy += dy
        
        # Create transform that keeps frame centered
        transform = np.array([
            [1, 0, -total_dx + center_offset_x],
            [0, 1, -total_dy + center_offset_y]
        ], dtype=np.float32)
        
        transforms.append(transform)
    
    return transforms

def compensate_motion(frames, flows):
    h, w = frames[0].shape[:2]
    pad = int(max(h, w) * 0.5)  
    # pad_right = int(max(h, w) * 0.7)  
    
    padded_frames = [cv2.copyMakeBorder(
        frame, pad, pad, pad, pad,
        cv2.BORDER_REPLICATE
    ) for frame in frames]

    # Calculate original path
    positions_x = [0]
    positions_y = [0]
    
    for flow in flows:
        dx = np.mean(flow[:, :, 0])
        dy = np.mean(flow[:, :, 1])
        positions_x.append(positions_x[-1] + dx)
        positions_y.append(positions_y[-1] + dy)
    
    positions_x = np.array(positions_x)
    positions_y = np.array(positions_y)
    
    # Find smooth path using optimization
    from scipy.optimize import minimize
    
    def objective(smooth_path, original_path):
        # Minimize both deviation from original path and acceleration
        smoothness_weight = 200.0
        
        # Deviation from original path
        path_error = np.sum((smooth_path - original_path) ** 2)
        
        # Acceleration (second derivative)
        acceleration = smooth_path[2:] - 2 * smooth_path[1:-1] + smooth_path[:-2]
        smoothness_error = np.sum(acceleration ** 2)
        
        return path_error + smoothness_weight * smoothness_error
    
    # Optimize x and y paths separately
    smooth_x = minimize(lambda x: objective(x, positions_x), positions_x.copy(), method='BFGS').x
    smooth_y = minimize(lambda x: objective(x, positions_y), positions_y.copy(), method='BFGS').x
    
    # Add initial offset
    smooth_x += pad * -0.1
    
    stabilized_frames = []
    for i in range(len(flows)):
        frame = padded_frames[i + 1]
        
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
        
        y_start = (h_pad - h) // 2
        x_start = (w_pad - w) // 2
        stabilized = stabilized[y_start:y_start+h, x_start:x_start+w]
        
        stabilized_frames.append(stabilized)
    
    # Add this to your compensate_motion function:
    plot = plot_motion_analysis(frames, flows, stabilized_frames)
    plot.savefig("motion_analysis.png")

    return stabilized_frames
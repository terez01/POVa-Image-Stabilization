import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def plot_motion(flows, stable_flows):
    """
    @brief Plot the camera motion over time.

    @param flows List of optical flow arrays.
    @param stable_flows List of stabilized optical flow arrays.

    @return None
    """
    stable_trajectory = []
    raw_trajectory = []

    # keep in mind the frame count decreases by stabilization 
    for flow in flows[:-1]:
        raw_dx = np.mean(flow[0]) 
        raw_dy = np.mean(flow[1]) 
        raw_trajectory.append((raw_dx, raw_dy))
    
    for stable_flow in stable_flows:
        stable_dx = np.mean(stable_flow[0])
        stable_dy = np.mean(stable_flow[1])
        stable_trajectory.append((stable_dx, stable_dy))
    
    x = np.arange(len(stable_trajectory))
    raw_x, raw_y = zip(*raw_trajectory)
    stable_x, stable_y = zip(*stable_trajectory)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(x, raw_y, label="Raw Vertical Motion", color="black", linestyle=":")
    plt.plot(x, stable_y, label="Stabilized Vertical Motion", color="black", linestyle="-")
    plt.xlabel("Frame Number")
    plt.ylabel("Vertical Displacement (pixels)")
    plt.title("Vertical Camera Motion Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x, raw_x, label="Raw Horizontal Motion", color="black", linestyle=":")
    plt.plot(x, stable_x, label="Stabilized Horizontal Motion", color="black", linestyle="-")
    plt.xlabel("Frame Number")
    plt.ylabel("Horizontal Displacement (pixels)")
    plt.title("Horizontal Camera Motion Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # can be switched to plt.show()
    plt.savefig("stabilization_plot.png")

def evaluate_stabilization(original_frames, stabilized_frames, original_flows, stabilized_flows):
    """
    @brief Evaluates stabilization by comparing original and stabilized sequences
    
    @param original_frames List of video frames before stabilization
    @param stabilized_frames List of video frames after stabilization
    @param original_flows List of optical flow arrays before stabilization
    @param stabilized_flows List of optical flow arrays after stabilization
    @return Dictionary containing comparison metrics
    """
    
    metrics = {}
    
    # flow mgnitude analysis
    def calculate_flow_magnitudes(flows):
        magnitudes = []
        for flow in flows:
            dx = np.mean(flow[:, :, 0])
            dy = np.mean(flow[:, :, 1])
            mag = np.sqrt(dx**2 + dy**2)
            magnitudes.append(mag)
        return np.array(magnitudes)
    
    orig_magnitudes = calculate_flow_magnitudes(original_flows)
    stab_magnitudes = calculate_flow_magnitudes(stabilized_flows)
    
    metrics['original_mean_motion'] = np.mean(orig_magnitudes)
    metrics['stabilized_mean_motion'] = np.mean(stab_magnitudes)
    
    # frame-to-frame similarity
    orig_ssim = []
    stab_ssim = []
    for i in range(len(original_frames)-1):
        orig_score = ssim(original_frames[i], original_frames[i+1], 
                         multichannel=True)
        stab_score = ssim(stabilized_frames[i], stabilized_frames[i+1], 
                         multichannel=True)
        orig_ssim.append(orig_score)
        stab_ssim.append(stab_score)
    
    metrics['original_stability'] = np.mean(orig_ssim)
    metrics['stabilized_stability'] = np.mean(stab_ssim)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    frames = np.arange(len(orig_magnitudes))
    plt.plot(frames, orig_magnitudes, 'r-', label='Original', alpha=0.7)
    plt.plot(frames, stab_magnitudes, 'b-', label='Stabilized', alpha=0.7)
    plt.title('Flow Magnitudes Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Motion Magnitude')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    frames = np.arange(len(orig_ssim))
    plt.plot(frames, orig_ssim, 'r-', label='Original', alpha=0.7)
    plt.plot(frames, stab_ssim, 'b-', label='Stabilized', alpha=0.7)
    plt.title('Frame-to-Frame Similarity (SSIM)')
    plt.xlabel('Frame')
    plt.ylabel('SSIM Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('motion_analysis.png')
    plt.close()
    
    return metrics

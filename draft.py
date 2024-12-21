import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from tqdm import tqdm

video_path = 'input/01_short.mp4'
print(f"Načítavam video: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Chyba: Video sa nepodarilo načítať.")
    exit()

# Read all frames first
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Convert frames to tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raft_model = raft_large(pretrained=True).eval().to(device)
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Prepare frames for optical flow computation
frame_tensors = [transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device) for frame in frames]

# Compute optical flow for all frame pairs in batch
flows = []
for i in tqdm(range(len(frame_tensors) - 1), desc="Calculating Optical Flow", ncols=100, unit="frame"):
    prev_tensor = frame_tensors[i]
    curr_tensor = frame_tensors[i + 1]

    with torch.no_grad():
        flow = raft_model(prev_tensor, curr_tensor)[-1][0].cpu().numpy()
    
    flows.append(flow)

# Define helper functions for affine transformation and stabilization
def estimate_affine_transform(flow):
    
    # Create a grid of coordinates
    h, w = flow.shape[1:3]
    y, x = np.mgrid[0:h, 0:w]
    coords_src = np.stack((x.ravel(), y.ravel()), axis=1)  # Source grid
    coords_dst = coords_src + flow.reshape(2, -1).T       # Destination points after flow

    # Filter valid points (ignore extreme flow values and outliers)
    valid_idx = (np.abs(flow[0]) < 30) & (np.abs(flow[1]) < 30)

    flow_mag = np.sqrt(flow[0]**2 + flow[1]**2)
    flow_median = np.median(flow_mag)
    flow_std = np.std(flow_mag)
    valid_idx = valid_idx & (flow_mag < (flow_median + 2 * flow_std))

    coords_src = coords_src[valid_idx.ravel()]
    coords_dst = coords_dst[valid_idx.ravel()]

    # Solve for affine transformation using least squares
    def residual(params):
        a, b, c, d, e, f = params
        A = np.array([[a, b, c], [d, e, f]])
        src_homo = np.hstack([coords_src, np.ones((coords_src.shape[0], 1))])
        dst_pred = src_homo @ A.T
        return (dst_pred[:, :2] - coords_dst).ravel()

    initial_guess = [1, 0, 0, 0, 1, 0]  # Identity transformation
    res = least_squares(residual, initial_guess, loss='soft_l1')
    affine_params = res.x
    matrix = np.array([[affine_params[0], affine_params[1], affine_params[2]],
                     [affine_params[3], affine_params[4], affine_params[5]]], dtype=np.float32)
    return np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2]


def smooth_affine_transform(transform, prev_transform=None, alpha=0.1):
    if prev_transform is None:
        return transform
    return prev_transform * alpha + transform * (1-alpha)

def stabilize_center(frame, transform, target_size, padding=50, zoom_factor=1.1):
     # Add padding to the frame
    h, w = frame.shape[:2]
    padded_frame = cv2.copyMakeBorder(frame, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    # Update dimensions after padding
    padded_h, padded_w = padded_frame.shape[:2]

    # Update the transformation matrix to account for padding
    padded_transform = transform.copy()
    padded_transform[0, 2] += padding
    padded_transform[1, 2] += padding

    # Apply affine transformation
    stabilized_frame = cv2.warpAffine(padded_frame, padded_transform, (padded_w, padded_h))

    # Crop with zoom factor
    center_x, center_y = padded_w // 2, padded_h // 2
    crop_w, crop_h = int(target_size[0] / zoom_factor), int(target_size[1] / zoom_factor)
    x1 = max(center_x - crop_w // 2, 0)
    y1 = max(center_y - crop_h // 2, 0)
    x2 = min(center_x + crop_w // 2, padded_w)
    y2 = min(center_y + crop_h // 2, padded_h)

    cropped_frame = stabilized_frame[y1:y2, x1:x2]

    # Resize back to target size
    resized_frame = cv2.resize(cropped_frame, target_size)
    return resized_frame

# Set up video writer
h, w = frames[0].shape[:2]
output_path = 'output/stabilizovane_video_raft.avi'
fps = 30  # Use your video's FPS
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (w, h)
)

# Variables to store trajectory for plotting
raw_trajectory = []
smoothed_trajectory = []
stabilized_frames =[]

# Now apply transformations on each frame
last_transform = None
for frame_idx in tqdm(range(len(frames) - 1), desc="Processing Frames", ncols=100, unit="frame"):
    curr_frame = frames[frame_idx + 1]
    flow = flows[frame_idx]
    
    raw_dx, raw_dy = np.mean(flow[0]), np.mean(flow[1])
    raw_trajectory.append((raw_dx, raw_dy))

    raw_transform = estimate_affine_transform(flow)
    smoothed_transform = smooth_affine_transform(raw_transform, last_transform, alpha=0.3)

    last_transform = smoothed_transform

    stabilized_frame = stabilize_center(curr_frame, smoothed_transform, (w, h))
    stabilized_frames.append(stabilized_frame)
    # Save the stabilized frame
    out.write(stabilized_frame)

# Release resources
out.release()

stabilized_tensors = [transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device) for frame in stabilized_frames]
stabilized_flows = []
for i in tqdm(range(len(stabilized_tensors) - 1), desc="Calculating New Optical Flow", ncols=100, unit="frame"):
    prev_tensor = stabilized_tensors[i]
    curr_tensor = stabilized_tensors[i + 1]

    with torch.no_grad():
        stabilized_flow = raft_model(prev_tensor, curr_tensor)[-1][0].cpu().numpy()
    
    stabilized_flows.append(stabilized_flow)

for frame_idx in tqdm(range(len(stabilized_frames) - 1), desc="Evaluating Frames", ncols=100, unit="frame"):
    curr_frame = frames[frame_idx + 1]
    stable_flow = stabilized_flows[frame_idx]
    
    stable_dx, stable_dy = np.mean(stable_flow[0]), np.mean(stable_flow[1])
    smoothed_trajectory.append((stable_dx, stable_dy))


# Create motion analysis plots
x = np.arange(len(smoothed_trajectory))
raw_x, raw_y = zip(*raw_trajectory)
smooth_x, smooth_y = zip(*smoothed_trajectory)
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(x, raw_y, label="Raw Vertical Motion", color="gray", linestyle=":")
plt.plot(x, smooth_y, label="Smoothed Vertical Motion", color="black", linestyle="-")
plt.xlabel("Frame Number")
plt.ylabel("Vertical Displacement (pixels)")
plt.title("Vertical Camera Motion Over Time")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(x, raw_x, label="Raw Horizontal Motion", color="gray", linestyle=":")
plt.plot(x, smooth_x, label="Smoothed Horizontal Motion", color="black", linestyle="-")
plt.xlabel("Frame Number")
plt.ylabel("Horizontal Displacement (pixels)")
plt.title("Horizontal Camera Motion Over Time")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("motion_analysis.png")

print(f"[INFO] Video saved to: {output_path}")

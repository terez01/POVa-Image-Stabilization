import cv2
from tqdm import tqdm

from torchvision.utils import flow_to_image

from conversion_utils import flows_to_tensors

def save_video(frames, output_path, fps, dimensions):
    """
    Saves video from frames
    """
    # CODEC
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_path, fourcc, fps, dimensions)

    for frame in tqdm(frames, desc="Saving Video", dynamic_ncols=True):
        out.write(frame)

    out.release()


def extract_frames(video_path):
    """
    @brief Extracts frames, FPS, and resolution from a video file.

    @param video_path Path to the video file.

    @return A tuple (frames,  fps, (width, height)):
        - frames: List of processed video frames.
        - fps: Video frames per second.
        - (width, height): Video resolution.

    @throws ValueError If the video file cannot be opened, dimensions are too small, or not divisible by 8.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    # Video properties
    total_number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolution constraints
    if width < 128 or height < 128:
        raise ValueError(f"Video resolution too small: {width}x{height}. Minimum size is 128x128.")
    if width % 8 != 0 or height % 8 != 0:
        raise ValueError(f"Video resolution {width}x{height} must be divisible by 8.")

    frames = []

    for _ in tqdm(range(total_number_of_frames), desc="Extracting frames", dynamic_ncols=True):
        # Read frame by frame
        ret, frame = cap.read()

        # No more frames to read
        if not ret:
            break   

        frames.append(frame)

    cap.release()

    return frames, fps, (width, height)


def visualize_flow(flows):
    """
    @brief Converts optical flow data to visualized frames.

    @param flows List of optical flow data.

    @return List of frames visualizing the optical flow.
    """
    flow_tensors = flows_to_tensors(flows)

    visualized_frames = []

    for flow_tensor in tqdm(flow_tensors, desc="Visualizing Optical Flow", dynamic_ncols=True):
        vis_frame = flow_to_image(flow_tensor)

        # Convert back to NumPy for visualization consistency
        vis_frame_np = vis_frame.permute(1, 2, 0).byte().numpy()  # Shape (H, W, 3)
        visualized_frames.append(vis_frame_np)

    return visualized_frames
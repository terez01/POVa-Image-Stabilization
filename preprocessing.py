import cv2
from tqdm import tqdm


def resize_frame(frame, width, height):
    """
    Resize a frame to the given dimensions.

    Args:
        frame (ndarray): The input frame to resize.
        width (int): The desired width of the resized frame.
        height (int): The desired height of the resized frame.

    Returns:
        ndarray: The resized frame.
    """
    return cv2.resize(frame, (width, height))


def convert_to_grayscale(frame):
    """
    Convert a frame to grayscale.

    Args:
        frame (ndarray): The input frame to convert.

    Returns:
        ndarray: The grayscale frame.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def adjust_dimensions(width, height):
    """
    Ensure that the width and height are divisible by 32. (because of the SEA-RAFT)

    Args:
        width (int): The width of the frame.
        height (int): The height of the frame.

    Returns:
        tuple: The adjusted width and height.
    """
    # Ensure dimensions are divisible by 32
    if width % 32 != 0 or height % 32 != 0:
        width = (width // 32) * 32
        height = (height // 32) * 32
    
    return width, height


def preprocess_video(video_path, scale_factor=1.0):
    """
    Load and preprocess the video, including resizing, converting to grayscale,
    and handling video properties like frame rate and dimensions.

    Args:
        video_path (str): Path to the video file to preprocess.
        scale_factor (float, optional): Factor by which to scale the video dimensions.
            Defaults to 1.0 (no scaling).

    Returns:
        tuple: A tuple containing:
            - frames (list): List of processed (grayscale and resized) frames.
            - original_frames (list): List of resized original color frames.
            - fps (float): Frames per second of the video.
            - dimensions (tuple): The final width and height of the frames.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")

    frames = []  # Processed (grayscale) frames
    original_frames = []  # Original color frames

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Apply scale factor to the original width and height
    width = int(width * scale_factor)
    height = int(height * scale_factor)

    # Adjust dimensions to be divisible by 32
    width, height = adjust_dimensions(width, height)

    print(f"Preprocessed Video Dimensions: {width}x{height}, FPS: {fps}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in tqdm(range(total_frames), desc="(Pre)processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        resized_original_frame = resize_frame(frame, width, height)
        original_frames.append(resized_original_frame)

        gray_frame = convert_to_grayscale(frame)
        resized_frame = resize_frame(gray_frame, width, height)
        frames.append(resized_frame)

    cap.release()
    return frames, original_frames, fps, (width, height)

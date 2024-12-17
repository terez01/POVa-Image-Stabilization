import cv2
from tqdm import tqdm


def play_video_from_path(video_path):
    """
    Plays a video from a file path.

    Args:
        video_path (str): Path to the video file to be played.

    Notes:
        - Press 'q' during playback to quit early.
        - Displays the video in a window named "Video Playback".
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            #! keep the print or add the bar?
            # print("End of video")
            break

        cv2.imshow("Video Playback", frame)

        # Exit playback if 'q' is pressed
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def play_video_from_frames(frames, fps):
    """
    Plays a video from a list of frames.

    Args:
        frames (list): List of frames (numpy arrays) to be played.
        fps (int): Frames per second for the video playback.

    Notes:
        - Press 'q' during playback to quit early.
        - Displays the video in a window named "Video Playback".
    """
    for frame in frames:
        cv2.imshow("Video Playback", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def save_video_from_frames(frames, output_path, fps, dimensions):
    """
    Saves a video from a list of frames.

    Args:
        frames (list): List of frames (numpy arrays) to be saved as a video.
        output_path (str): Path to save the output video file.
        fps (int): Frames per second for the output video.
        dimensions (tuple): Dimensions of the video frames (width, height).

    Notes:
        - Uses the 'XVID' codec for video compression.
        - Displays a progress bar while saving frames.
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec
    out = cv2.VideoWriter(output_path, fourcc, fps, dimensions)

    for frame in tqdm(frames, desc="Saving Video"):
        out.write(frame)

    out.release()
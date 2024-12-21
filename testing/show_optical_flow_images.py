import cv2
import os

# List of video files (assumes all three optical flow videos are present)
farneback_path = "../output/visualized_flow_farneback.avi"
deepflow_path = "../output/visualized_flow_deepflow.avi"
raft_path = "../output/visualized_flow_raft.avi"

video_files = [farneback_path, deepflow_path, raft_path]
output_images = ["frame_farneback.jpg", "frame_deepflow.jpg", "frame_raft.jpg"]
output_size = (960, 520)  # Width, Height

# Directory for saving output images
output_directory = "out_vis_flow"

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

for video_file, output_image in zip(video_files, output_images):
    # Open the video file
    cap = cv2.VideoCapture(video_file)
    
    if not cap.isOpened():
        print(f"Error opening video file: {video_file}")
        continue

    # Read the first frame
    ret, frame = cap.read()
    if ret:
        # Resize the frame
        resized_frame = cv2.resize(frame, output_size)

        # Build the full output path
        output_path = os.path.join(output_directory, output_image)
        
        # Save the resized frame
        cv2.imwrite(output_path, resized_frame)
        print(f"Saved {output_path}")
    else:
        print(f"Could not read frame from {video_file}")
    
    cap.release()

import os
import sys
from eval_utils import compute_endpoint_error, read_flo_file, load_images

# Add the parent directory to the sys.path to import optical_flow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optical_flow import estimate_optical_flow_raft, estimate_optical_flow_deepflow, estimate_optical_flow_farneback


def crop_image(image):
    """Function crops image (2 pixels from top and bottom)."""
    return image[2:-2, :]


def crop_flow(flow):
    """Function crops flow (2 pixels from top and bottom)."""
    return flow[2:-2, :, :]


def process_sequence(sequence_dir):
    """Function processes a sequence of frames"""

    epe_local_totals = {'raft': 0, 'farneback': 0, 'deepflow': 0}
    frame_count = 0

    # Paths for the current sequence's flow and image directories
    sequence_flow_dir = os.path.join(flow_dir, sequence_dir)
    sequence_image_dir = os.path.join(image_dir, sequence_dir)

    # List image and flow files
    image_files = sorted(os.listdir(sequence_image_dir))
    flow_files = sorted(os.listdir(sequence_flow_dir))

    # Ensure the number of flow files corresponds the number of image frames - 1
    assert len(flow_files) == len(image_files) - 1, f"Flow files mismatch for sequence: {sequence_dir}"

    # Process each frame pair
    for i in range(len(flow_files)):
        frame_1_image = image_files[i]
        frame_2_image = image_files[i + 1]
        flow_file = flow_files[i]

        # Read the ground truth flow for this frame pair
        flow_gt = read_flo_file(os.path.join(sequence_flow_dir, flow_file))

        # Crop the flow to match the image crop
        cropped_flow_gt = crop_flow(flow_gt)

        img1_path = os.path.join(sequence_image_dir, frame_1_image)
        img2_path = os.path.join(sequence_image_dir, frame_2_image)

        img1, img2 = load_images(img1_path, img2_path)

        # Images need to be cropped in order to have dimensions divisible by 8 
        cropped_img1 = crop_image(img1)
        cropped_img2 = crop_image(img2)

        # Estimate flow using RAFT, Farneback, and DeepFlow
        flow_pred_raft = estimate_optical_flow_raft([cropped_img1, cropped_img2])[0]
        flow_pred_farneback = estimate_optical_flow_farneback([cropped_img1, cropped_img2])[0]
        flow_pred_deepflow = estimate_optical_flow_deepflow([cropped_img1, cropped_img2])[0]

        # Compute EPE for each method
        epe_raft = compute_endpoint_error(cropped_flow_gt, flow_pred_raft)
        epe_farneback = compute_endpoint_error(cropped_flow_gt, flow_pred_farneback)
        epe_deepflow = compute_endpoint_error(cropped_flow_gt, flow_pred_deepflow)

        # Accumulate errors locally
        epe_local_totals['raft'] += epe_raft
        epe_local_totals['farneback'] += epe_farneback
        epe_local_totals['deepflow'] += epe_deepflow
        frame_count += 1

        with open(output_file, "a") as file:
            file.write(f"{flow_file} | {epe_raft:.4f} | {epe_farneback:.4f} | {epe_deepflow:.4f}\n")

    return epe_local_totals, frame_count


# Define paths to the MPI-Sintel dataset
flow_dir = "/mnt/c/Users/Acer/Documents/POVa datasets/MPI-Sintel-complete/training/flow"
image_dir = "/mnt/c/Users/Acer/Documents/POVa datasets/MPI-Sintel-complete/training/final"
output_file = "out_eval/epe_results_sintel.txt"

# Get all sequence directories (e.g., 'alley_1', 'alley_2', ...)
sequence_dirs = os.listdir(flow_dir)

# Accumulators for EPE values
epe_totals = {'raft': 0, 'farneback': 0, 'deepflow': 0}
total_frames = 0  # Total frame count will be accumulated here

with open(output_file, "w") as file:
    file.write("Frame | RAFT EPE | Farneback EPE | DeepFlow EPE\n")
    file.write("-" * 50 + "\n")

# Loop through all sequences and process each one
for seq_dir in sequence_dirs:
    print(f"Processing sequence: {seq_dir}")
    epe_totals_sequence, frame_count_sequence = process_sequence(seq_dir)

    # Accumulate the total errors and frame counts
    for method in epe_totals:
        epe_totals[method] += epe_totals_sequence[method]
    total_frames += frame_count_sequence

# Calculate average EPE for each method
epe_avg = {key: total / total_frames for key, total in epe_totals.items()}

with open(output_file, "a") as file:
    file.write("\nAverage Results:\n")
    file.write(f"RAFT Average EPE: {epe_avg['raft']:.4f}\n")
    file.write(f"Farneback Average EPE: {epe_avg['farneback']:.4f}\n")
    file.write(f"DeepFlow Average EPE: {epe_avg['deepflow']:.4f}\n")

print(f"Results saved to {output_file}")
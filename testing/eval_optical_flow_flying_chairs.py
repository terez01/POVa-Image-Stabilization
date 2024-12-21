import os
import sys
from eval_utils import compute_endpoint_error, read_flo_file, load_images

# Add the parent directory to the sys.path to import optical_flow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optical_flow import estimate_optical_flow_raft, estimate_optical_flow_deepflow, estimate_optical_flow_farneback


# Path to the FlyingChairs dataset
data_dir = "/mnt/c/Users/Acer/Documents/POVa datasets/FlyingChairs/FlyingChairs_release/data"
output_file = "out_eval/epe_results_flying_chairs.txt"

# limit to 1000 due to HW constraints
flow_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_flow.flo')])[:1000]

epe_totals = {'raft': 0, 'farneback': 0, 'deepflow': 0}
frame_count = 0

with open(output_file, "w") as file:
    file.write("Frame | RAFT EPE | Farneback EPE | DeepFlow EPE\n")
    file.write("-" * 50 + "\n")

    for flow_file in flow_files:
        flow_path = os.path.join(data_dir, flow_file)
        img1_file = flow_file.replace('_flow.flo', '_img1.ppm')
        img2_file = flow_file.replace('_flow.flo', '_img2.ppm')
        img1_path = os.path.join(data_dir, img1_file)
        img2_path = os.path.join(data_dir, img2_file)

        img1, img2 = load_images(img1_path, img2_path)

        # Read ground truth flow
        flow_gt = read_flo_file(flow_path)

        # Estimate flow using RAFT, Farneback, and DeepFlow
        flow_pred_raft = estimate_optical_flow_raft([img1, img2])[0]
        flow_pred_farneback = estimate_optical_flow_farneback([img1, img2])[0]
        flow_pred_deepflow = estimate_optical_flow_deepflow([img1, img2])[0]

        # Compute EPE for each method
        epe_raft = compute_endpoint_error(flow_gt, flow_pred_raft)
        epe_farneback = compute_endpoint_error(flow_gt, flow_pred_farneback)
        epe_deepflow = compute_endpoint_error(flow_gt, flow_pred_deepflow)

        # Accumulate the errors
        epe_totals['raft'] += epe_raft
        epe_totals['farneback'] += epe_farneback
        epe_totals['deepflow'] += epe_deepflow
        frame_count += 1

        file.write(f"{flow_file} | {epe_raft:.4f} | {epe_farneback:.4f} | {epe_deepflow:.4f}\n")

    # Compute average EPE
    epe_avg = {key: total / frame_count for key, total in epe_totals.items()}

    file.write("\nAverage Results:\n")
    file.write(f"RAFT Average EPE: {epe_avg['raft']:.4f}\n")
    file.write(f"Farneback Average EPE: {epe_avg['farneback']:.4f}\n")
    file.write(f"DeepFlow Average EPE: {epe_avg['deepflow']:.4f}\n")

print(f"Results saved to {output_file}")
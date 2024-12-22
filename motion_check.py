import numpy as np
import matplotlib.pyplot as plt


def plot_motion(flows, stable_flows):
    stable_trajectory = []
    raw_trajectory = []

    for flow in flows[:-1]:
        raw_dx, raw_dy = np.mean(flow[0]), np.mean(flow[1])
        raw_trajectory.append((raw_dx, raw_dy))

    for stable_flow in stable_flows:
        stable_dx, stable_dy = np.mean(stable_flow[0]), np.mean(stable_flow[1])
        stable_trajectory.append((stable_dx, stable_dy))
    
    x = np.arange(len(stable_trajectory))
    raw_x, raw_y = zip(*raw_trajectory)
    stable_x, stable_y = zip(*stable_trajectory)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    plt.plot(x, raw_y, label="Raw Vertical Motion", color="gray", linestyle=":")
    plt.plot(x, stable_y, label="Smoothed Vertical Motion", color="black", linestyle="-")
    plt.xlabel("Frame Number")
    plt.ylabel("Vertical Displacement (pixels)")
    plt.title("Vertical Camera Motion Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x, raw_x, label="Raw Horizontal Motion", color="gray", linestyle=":")
    plt.plot(x, stable_x, label="Smoothed Horizontal Motion", color="black", linestyle="-")
    plt.xlabel("Frame Number")
    plt.ylabel("Horizontal Displacement (pixels)")
    plt.title("Horizontal Camera Motion Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("motion_analysis.png")

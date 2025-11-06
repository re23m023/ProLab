#!/usr/bin/env python3
# =========================================================
# This script automatically analyzes all .bag files
# located in the ./bags/ directory. It:
#   ✅ extracts Ground Truth, KF, EKF, and PF positions
#   ✅ computes RMSE (x, y)
#   ✅ saves CSV results and trajectory plots
#   ✅ generates a summary table for all runs
#
# Usage in Terminal:
# ---------------------------------------------------------
# 1. Start simulation and record runs:
#    cd ~/catkin_pro/src/example_package/bags
#    rosbag record -O run1.bag /odom /imu /gazebo/model_states \
#        /prediction_KF /prediction_EKF /prediction_particle \
#        /prediction_particle_mean /cmd_vel /scan
#
# 2. After recording several runs:
#    cd ~/catkin_pro/src/example_package
#    python3 ./scripts/analyse_tool.py
#
# 3. You will find the results here:
#    ~/catkin_pro/src/example_package/results/
#       ├── run1_rmse.csv
#       ├── run1_traj.png
#       ├── ...
#       └── summary_rmse_all_runs.csv
# =========================================================

import rosbag
import numpy as np
import matplotlib.pyplot as plt
import os, glob, csv
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray

# =========================================================
# Helper Functions
# =========================================================
def rmse(a, b):
    """Compute Root Mean Square Error between two equally long arrays."""
    return np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2))

def extract_pose(msg):
    """Extract (x, y) position from a PoseWithCovarianceStamped message."""
    return [msg.pose.pose.position.x, msg.pose.pose.position.y]

def extract_ground_truth(msg):
    """Extract ground truth position from gazebo/model_states message."""
    idx = -1
    for name in msg.name:
        if "turtlebot3" in name:  # works for turtlebot3_burger / waffle
            idx = msg.name.index(name)
            break
    if idx != -1:
        p = msg.pose[idx].position
        return [p.x, p.y]
    return None

def analyze_bag(bagfile):
    """Read a rosbag file and extract GT, KF, EKF, and PF positions."""
    bag = rosbag.Bag(bagfile)
    gt, kf, ekf, pf = [], [], [], []

    for topic, msg, t in bag.read_messages():
        # --- Ground Truth from Gazebo ---
        if topic == "/gazebo/model_states":
            p = extract_ground_truth(msg)
            if p:
                gt.append(p)

        # --- Kalman Filter prediction ---
        elif topic == "/prediction_KF":
            kf.append(extract_pose(msg))

        # --- Extended Kalman Filter prediction ---
        elif topic == "/prediction_EKF":
            ekf.append(extract_pose(msg))

        # --- Particle Filter mean position ---
        elif topic == "/prediction_pf_mean":
            pf.append(extract_pose(msg))

    bag.close()
    return np.array(gt), np.array(kf), np.array(ekf), np.array(pf)

# =========================================================
# Main Analysis
# =========================================================
if __name__ == "__main__":
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # --- Locate all bag files in ./bags/ ---
    bag_files = sorted(glob.glob("./bags/*.bag"))
    if not bag_files:
        print("No .bag files found in ./bags/ directory.")
        exit(1)

    summary_rows = []

    for bagfile in bag_files:
        print(f"\nAnalyzing: {bagfile}")
        gt, kf, ekf, pf = analyze_bag(bagfile)

        # --- Ensure equal length arrays ---
        n = min(len(gt), len(kf), len(ekf), len(pf))
        if n == 0:
            print("No valid data found! Skipping file.")
            continue
        gt, kf, ekf, pf = gt[:n], kf[:n], ekf[:n], pf[:n]

        # --- Compute RMSE for each filter ---
        rmse_kf_x = rmse(gt[:, 0], kf[:, 0])
        rmse_kf_y = rmse(gt[:, 1], kf[:, 1])
        rmse_ekf_x = rmse(gt[:, 0], ekf[:, 0])
        rmse_ekf_y = rmse(gt[:, 1], ekf[:, 1])
        rmse_pf_x = rmse(gt[:, 0], pf[:, 0])
        rmse_pf_y = rmse(gt[:, 1], pf[:, 1])

        print(f"KF : ({rmse_kf_x:.3f}, {rmse_kf_y:.3f})  m")
        print(f"EKF: ({rmse_ekf_x:.3f}, {rmse_ekf_y:.3f}) m")
        print(f"PF : ({rmse_pf_x:.3f}, {rmse_pf_y:.3f})  m")

        # --- Store results for summary ---
        summary_rows.append([
            os.path.basename(bagfile),
            rmse_kf_x, rmse_kf_y,
            rmse_ekf_x, rmse_ekf_y,
            rmse_pf_x, rmse_pf_y
        ])

        # --- Save CSV for this run ---
        csv_name = os.path.join(
            results_dir,
            os.path.basename(bagfile).replace(".bag", "_rmse.csv")
        )
        with open(csv_name, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filter", "rmse_x_m", "rmse_y_m"])
            w.writerow(["KF", rmse_kf_x, rmse_kf_y])
            w.writerow(["EKF", rmse_ekf_x, rmse_ekf_y])
            w.writerow(["PF", rmse_pf_x, rmse_pf_y])
        print(f"Results saved: {csv_name}")

        # --- Optional: Plot trajectory ---
        plt.figure(figsize=(6, 5))
        plt.plot(gt[:, 0], gt[:, 1], "k-", label="Ground Truth")
        plt.plot(kf[:, 0], kf[:, 1], "b--", label="KF")
        plt.plot(ekf[:, 0], ekf[:, 1], "g--", label="EKF")
        plt.plot(pf[:, 0], pf[:, 1], "r--", label="PF (Mean)")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"Trajectory - {os.path.basename(bagfile)}")
        plt.legend()
        plt.grid()
        traj_path = os.path.join(
            results_dir,
            os.path.basename(bagfile).replace(".bag", "_traj.png")
        )
        plt.savefig(traj_path, dpi=150)
        plt.close()

    # =========================================================
    # Save Overall Summary for All Runs
    # =========================================================
    summary_csv = os.path.join(results_dir, "summary_rmse_all_runs.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "bagfile",
            "KF_x", "KF_y",
            "EKF_x", "EKF_y",
            "PF_x", "PF_y"
        ])
        w.writerows(summary_rows)
    print(f"\n✅ Summary of all runs saved at:\n{summary_csv}")


"""
data.py
---------------------------------
Generate a synthetic dataset for Lane Change Prediction.

Yeh dataset ek simple self-driving / driver-assist system ko simulate karta hai:
- Car ki speed
- Steering angle
- Left & right lane distance
- Road curvature (road kitna mod raha hai)
- Output action: KEEP_LANE / CHANGE_LEFT / CHANGE_RIGHT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_lane_data(n_samples=2500):
    """
    Generate a realistic-style lane change dataset.
    """
    print("🚗 Generating lane change dataset...")
    np.random.seed(42)

    # ------------------ BASIC FEATURES ------------------
    # Car speed (city + highway mix)
    speed = np.random.uniform(30, 120, n_samples)

    # Steering angle: 0 = straight, negative = left, positive = right
    steering = np.random.normal(0, 15, n_samples)

    # Base distance from lane markings (meters)
    base_dist = np.random.uniform(1.0, 2.5, n_samples)
    variation = np.random.normal(0, 0.5, n_samples)

    # Distance to left and right lane boundaries
    dist_left = np.clip(base_dist + variation, 0.2, 4.0)
    dist_right = np.clip(base_dist - variation, 0.2, 4.0)

    # Road curvature (negative = left bend, positive = right bend)
    road_curve = np.random.normal(0, 10, n_samples)

    # ------------------ LABEL GENERATION ------------------
    actions = []

    for i in range(n_samples):
        dl = dist_left[i]
        dr = dist_right[i]
        curve = road_curve[i]
        steer = steering[i]

        # Simple rule-based logic to create labels (ground truth)
        if dl < 0.7 and dr > 1.2:
            # Too close left, right side safer
            actions.append("CHANGE_RIGHT")
        elif dr < 0.7 and dl > 1.2:
            # Too close right, left side safer
            actions.append("CHANGE_LEFT")
        elif curve < -15 and dl > 1.0:
            # Sharp left curve, left lane may be better
            actions.append("CHANGE_LEFT")
        elif curve > 15 and dr > 1.0:
            # Sharp right curve, right lane may be better
            actions.append("CHANGE_RIGHT")
        else:
            actions.append("KEEP_LANE")

    # ------------------ DATAFRAME ------------------
    df = pd.DataFrame({
        "speed": speed,
        "steering_angle": steering,
        "dist_left": dist_left,
        "dist_right": dist_right,
        "road_curve": road_curve,
        "action": actions
    })

    # Save to CSV for training
    df.to_csv("lane_data.csv", index=False)

    print(f"\n✅ Dataset created with {n_samples} samples")
    print("\n📊 Class distribution:")
    print(df["action"].value_counts())

    # ------------------ SIMPLE VISUALIZATION ------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Speed distribution
    axes[0, 0].hist(df["speed"], bins=30, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Speed Distribution")
    axes[0, 0].set_xlabel("Speed (km/h)")

    # Steering angle
    axes[0, 1].hist(df["steering_angle"], bins=30, color="lightgreen", edgecolor="black")
    axes[0, 1].set_title("Steering Angle")
    axes[0, 1].set_xlabel("Angle (degrees)")

    # Lane distances
    axes[0, 2].boxplot([df["dist_left"], df["dist_right"]], labels=["Left", "Right"])
    axes[0, 2].set_title("Lane Distances")
    axes[0, 2].set_ylabel("Distance (m)")

    # Road curve
    axes[1, 0].hist(df["road_curve"], bins=30, color="lightcoral", edgecolor="black")
    axes[1, 0].set_title("Road Curvature")
    axes[1, 0].set_xlabel("Curve (degrees)")

    # Action distribution
    action_counts = df["action"].value_counts()
    colors = ["green", "blue", "red"]
    axes[1, 1].bar(action_counts.index, action_counts.values, color=colors)
    axes[1, 1].set_title("Action Distribution")
    axes[1, 1].set_ylabel("Count")

    # Safety margin (minimum distance to any lane marking)
    safety_margin = df[["dist_left", "dist_right"]].min(axis=1)
    axes[1, 2].hist(safety_margin, bins=30, color="gold", edgecolor="black")
    axes[1, 2].set_title("Minimum Lane Distance")
    axes[1, 2].set_xlabel("Distance (m)")

    plt.tight_layout()
    plt.savefig("dataset_distribution.png", dpi=100, bbox_inches="tight")
    print("📈 Visualization saved as 'dataset_distribution.png'")

    return df


if __name__ == "__main__":
    # Generate dataset
    data = generate_lane_data(2500)

    print("\n📋 Dataset preview:")
    print(data.head())
    print("\n📊 Basic statistics:")
    print(data.describe())

#!/usr/bin/env python3
"""
Environment analysis script.

This script analyzes the properties of the obstacle configurations
and generates statistics and visualizations of the distributions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loader import load_environments

def analyze_environments(env_file="environments/env_configurations.json", output_dir="figures/analysis"):
    """
    Analyze the environments in the JSON file and generate statistics.
    
    Args:
        env_file: Path to the JSON file with environment configurations
        output_dir: Directory to save the analysis visualizations
    """
    # Load environments
    try:
        environments = load_environments(env_file)
        print(f"Loaded {len(environments)} environments from {env_file}")
    except Exception as e:
        print(f"Error loading environments: {e}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for analysis
    data = []
    for env in environments:
        env_id = env.get("id", "unknown")
        obstacles = env.get("obstacles", [])
        start = env.get("start", (1, 1))
        goal = env.get("goal", (9, 9))
        
        # Calculate obstacle area and other metrics
        total_obstacle_area = sum(obs[2] * obs[3] for obs in obstacles)  # w * h
        num_obstacles = len(obstacles)
        avg_obstacle_size = total_obstacle_area / num_obstacles if num_obstacles > 0 else 0
        
        # Calculate distance from start to goal
        start_goal_distance = np.linalg.norm(np.array(start) - np.array(goal))
        
        # Store data
        data.append({
            "env_id": env_id,
            "num_obstacles": num_obstacles,
            "total_obstacle_area": total_obstacle_area,
            "avg_obstacle_size": avg_obstacle_size,
            "start_goal_distance": start_goal_distance,
            "start_x": start[0],
            "start_y": start[1],
            "goal_x": goal[0],
            "goal_y": goal[1]
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(data)
    
    # Basic statistics
    stats = df.describe()
    print("\n=== Environment Statistics ===")
    print(stats)
    
    # Save statistics to CSV
    stats.to_csv(os.path.join(output_dir, "environment_stats.csv"))
    
    # Create visualizations
    
    # 1. Number of obstacles distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["num_obstacles"], bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.title("Distribution of Number of Obstacles")
    plt.xlabel("Number of Obstacles")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "num_obstacles_dist.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 2. Total obstacle area distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["total_obstacle_area"], bins=15, alpha=0.7, color='green', edgecolor='black')
    plt.title("Distribution of Total Obstacle Area")
    plt.xlabel("Total Obstacle Area")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "obstacle_area_dist.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 3. Average obstacle size distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["avg_obstacle_size"], bins=15, alpha=0.7, color='red', edgecolor='black')
    plt.title("Distribution of Average Obstacle Size")
    plt.xlabel("Average Obstacle Size")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "avg_obstacle_size_dist.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 4. Start-goal distance distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["start_goal_distance"], bins=15, alpha=0.7, color='purple', edgecolor='black')
    plt.title("Distribution of Start-Goal Distance")
    plt.xlabel("Start-Goal Distance")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, "start_goal_dist.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 5. Scatter plot of start and goal positions
    plt.figure(figsize=(10, 10))
    plt.scatter(df["start_x"], df["start_y"], c='green', marker='o', s=100, label='Start')
    plt.scatter(df["goal_x"], df["goal_y"], c='red', marker='x', s=100, label='Goal')
    for i, (sx, sy, gx, gy) in enumerate(zip(df["start_x"], df["start_y"], df["goal_x"], df["goal_y"])):
        plt.plot([sx, gx], [sy, gy], 'k--', alpha=0.2)
    plt.title("Start and Goal Positions")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.savefig(os.path.join(output_dir, "start_goal_positions.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    # 6. Correlation heatmap
    correlation = df.drop('env_id', axis=1).corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title("Correlation Between Environment Features")
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha='right')
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    
    # Add correlation values
    for i in range(len(correlation.columns)):
        for j in range(len(correlation.columns)):
            plt.text(j, i, f'{correlation.iloc[i, j]:.2f}', ha='center', va='center', 
                    color='black' if abs(correlation.iloc[i, j]) < 0.7 else 'white')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete. Visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Create analysis directory
    os.makedirs("figures/analysis", exist_ok=True)
    
    # Run analysis
    analyze_environments() 
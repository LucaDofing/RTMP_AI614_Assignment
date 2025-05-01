#!/usr/bin/env python3
"""
Environment visualization script.

This script loads the obstacle configurations from the JSON file and 
creates visualizations of each environment, saving them to a subfolder in figures.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Arrow
from loader import load_environments

def visualize_environment(env, output_path, show_title=True, dpi=100):
    """
    Visualize a single environment configuration and save it as a PNG file.
    
    Args:
        env: Environment dictionary with width, height, start, goal, obstacles, and id
        output_path: Path to save the visualization
        show_title: Whether to show the environment ID in the title
        dpi: Resolution of the output image
    """
    # Extract environment parameters
    width = env.get("width", 10)
    height = env.get("height", 10)
    start = env.get("start", (1, 1))
    goal = env.get("goal", (9, 9))
    obstacles = env.get("obstacles", [])
    env_id = env.get("id", "unknown")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    
    if show_title:
        ax.set_title(f"Environment: {env_id}")
    
    # Draw obstacles
    for obs in obstacles:
        x, y, w, h = obs
        ax.add_patch(Rectangle((x, y), w, h, color='gray', alpha=0.7))
    
    # Draw start and goal with directional indicators
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
    ax.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Add a small arrow pointing from start toward goal to indicate direction
    start_vec = np.array(start)
    goal_vec = np.array(goal)
    direction = goal_vec - start_vec
    direction = direction / np.linalg.norm(direction) * 0.5  # Scale to a fixed length
    
    ax.arrow(start[0], start[1], direction[0], direction[1], 
             head_width=0.2, head_length=0.3, fc='green', ec='green')
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    # Save the figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def visualize_all_environments(env_file="environments/env_configurations.json", 
                              output_dir="figures/environments", 
                              max_envs=None):
    """
    Visualize all environments from the JSON file.
    
    Args:
        env_file: Path to the JSON file with environment configurations
        output_dir: Directory to save the visualizations
        max_envs: Maximum number of environments to visualize (None for all)
    """
    # Load environments
    try:
        environments = load_environments(env_file)
        print(f"Loaded {len(environments)} environments from {env_file}")
    except Exception as e:
        print(f"Error loading environments: {e}")
        return
    
    # Limit number of environments if specified
    if max_envs is not None:
        environments = environments[:max_envs]
        print(f"Visualizing first {max_envs} environments")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize each environment
    for i, env in enumerate(environments):
        env_id = env.get("id", f"env_{i}")
        output_path = os.path.join(output_dir, f"{env_id}.png")
        visualize_environment(env, output_path)
        if i % 20 == 0:  # Print progress every 20 environments
            print(f"Visualized {i+1}/{len(environments)} environments")
    
    print(f"Successfully visualized {len(environments)} environments to {output_dir}")

def create_environment_grid(env_file="environments/env_configurations.json",
                           output_path="figures/environment_grid.png",
                           rows=5, cols=5, starting_idx=0):
    """
    Create a grid visualization of multiple environments.
    
    Args:
        env_file: Path to the JSON file with environment configurations
        output_path: Path to save the grid visualization
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        starting_idx: Index of the first environment to include in the grid
    """
    # Load environments
    environments = load_environments(env_file)
    
    # Create figure
    fig, axs = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    
    # Flatten axs array for easier indexing
    if rows == 1 and cols == 1:
        axs = np.array([axs])
    elif rows == 1:
        axs = axs.reshape(1, cols)
    elif cols == 1:
        axs = axs.reshape(rows, 1)
    
    # Visualize each environment in the grid
    for i in range(rows):
        for j in range(cols):
            env_idx = starting_idx + i * cols + j
            if env_idx < len(environments):
                env = environments[env_idx]
                ax = axs[i, j]
                
                # Extract environment parameters
                width = env.get("width", 10)
                height = env.get("height", 10)
                start = env.get("start", (1, 1))
                goal = env.get("goal", (9, 9))
                obstacles = env.get("obstacles", [])
                env_id = env.get("id", f"env_{env_idx}")
                
                # Setup axes
                ax.set_xlim(0, width)
                ax.set_ylim(0, height)
                ax.set_aspect('equal')
                ax.set_title(f"{env_id}", fontsize=10)
                
                # Draw obstacles
                for obs in obstacles:
                    x, y, w, h = obs
                    ax.add_patch(Rectangle((x, y), w, h, color='gray', alpha=0.7))
                
                # Draw start and goal
                ax.plot(start[0], start[1], 'go', markersize=6)
                ax.plot(goal[0], goal[1], 'ro', markersize=6)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.5)
            else:
                # Hide unused subplots
                axs[i, j].axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Created environment grid at {output_path}")

if __name__ == "__main__":
    # Create directories
    os.makedirs("figures", exist_ok=True)
    os.makedirs("figures/environments", exist_ok=True)
    
    # Visualize all environments individually
    visualize_all_environments()
    
    # Create grids of environments
    create_environment_grid(rows=5, cols=5, starting_idx=0, 
                            output_path="figures/environment_grid_1.png")
    create_environment_grid(rows=5, cols=5, starting_idx=25, 
                            output_path="figures/environment_grid_2.png") 
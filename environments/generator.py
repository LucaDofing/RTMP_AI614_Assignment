"""
Environment generator for pathfinding algorithm benchmarks.
Generates random obstacle configurations and saves them to JSON.
"""

import json
import random
import os
import numpy as np
import time
import sys
from itertools import product

# Add the parent directory to the Python path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import a planner for validation
try:
    from planners.part2_2_rrt_star_improved import run_rrt_star_improved as validate_planner
except ImportError:
    try:
        sys.path.append(parent_dir)
        from planners.part2_2_rrt_star_improved import run_rrt_star_improved as validate_planner
    except ImportError:
        print("Warning: Could not import planner for validation. Skipping solvability check.")
        validate_planner = None

def is_valid_obstacle(new_obstacle, existing_obstacles, start, goal, min_distance=0.5):
    """Check if a new obstacle is valid (doesn't block start/goal and doesn't overlap other obstacles)"""
    x, y, w, h = new_obstacle
    
    # Check if obstacle contains start or goal
    if (x <= start[0] <= x + w and y <= start[1] <= y + h) or \
       (x <= goal[0] <= x + w and y <= goal[1] <= y + h):
        return False
    
    # Check if obstacle is too close to start or goal
    if (abs(x - start[0]) < min_distance and abs(y - start[1]) < min_distance) or \
       (abs(x - goal[0]) < min_distance and abs(y - goal[1]) < min_distance):
        return False
    
    # Check overlap with existing obstacles
    for ox, oy, ow, oh in existing_obstacles:
        # Check if rectangles overlap
        if (x < ox + ow and x + w > ox and y < oy + oh and y + h > oy):
            return False
    
    return True

def generate_random_start_goal(width=10, height=10, min_distance=5.0):
    """Generate random start and goal positions with a minimum distance between them"""
    # Generate random positions
    # start at a random position
    start_x = random.uniform(0.5, width - 0.5)  # Start in the left third
    start_y = random.uniform(0.5, height - 0.5)
        
    while True:
        # goal with min distance of 5
        
        goal_x = random.uniform(0.5, width - 0.5)
        goal_y = random.uniform(0.5, height - 0.5)
        
        start = (start_x, start_y)
        goal = (goal_x, goal_y)
        
        # Check distance
        dist = np.linalg.norm(np.array(start) - np.array(goal))
        if dist >= min_distance:
            return start, goal

def generate_random_environment(width=10, height=10, start=None, goal=None, 
                                num_obstacles=3, min_obstacle_size=1, max_obstacle_size=3):
    """Generate a random environment with obstacles"""
    # Generate random start and goal if not provided
    if start is None or goal is None:
        start, goal = generate_random_start_goal(width, height)
    
    obstacles = []
    
    for _ in range(num_obstacles * 3):  # Try multiple times to place each obstacle
        if len(obstacles) >= num_obstacles:
            break
            
        # Generate random position and size
        x = random.uniform(0, width - min_obstacle_size)
        y = random.uniform(0, height - min_obstacle_size)
        w = random.uniform(min_obstacle_size, min(max_obstacle_size, width - x))
        h = random.uniform(min_obstacle_size, min(max_obstacle_size, height - y))
        
        new_obstacle = (x, y, w, h)
        
        if is_valid_obstacle(new_obstacle, obstacles, start, goal):
            obstacles.append(new_obstacle)
    
    return {
        "width": width,
        "height": height,
        "start": start,
        "goal": goal,
        "obstacles": obstacles
    }

def is_environment_solvable(env, max_validation_iters=5000):
    """Check if an environment is solvable using the validation planner"""
    if validate_planner is None:
        raise ValueError("Planner is not available")
    

    # Extract environment parameters
    width = env.get("width", 10)
    height = env.get("height", 10)
    start = env.get("start", (1, 1))
    goal = env.get("goal", (9, 9))
    obstacles = env.get("obstacles", [])
    
    # Run the planner with high iterations and timeout
    try:
        result = validate_planner(
            width=width,
            height=height,
            start=start,
            goal=goal,
            obstacles=obstacles,
            max_iters=max_validation_iters,
            benchmark_mode=True,
            goal_threshold=0.5,
            step_size=0.5,  # Smaller step size for more precision
            goal_sample_rate=0.2  # Higher goal bias
        )
        return result.get("success", False)
    except Exception as e:
        print(f"Validation error: {e}")
        return False

def generate_environment_set(num_environments=200, output_file="environments/env_configurations.json",
                            validate_solvability=True, max_validation_iters=5000):
    """Generate a set of random environments and save to JSON"""
    environments = []
    
    # Create default environment
    default_env = {
        "id": "default",
        "width": 10,
        "height": 10,
        "start": (1, 1),
        "goal": (9, 9),
        "obstacles": [
            (3, 3, 2, 1.5),
            (6, 5, 1, 3),
            (2, 7, 2, 1)
        ]
    }
    environments.append(default_env)
    
    # Generate random environments
    print(f"Generating {num_environments - 1} random environments...")
    i = 0
    validation_failures = 0
    while i < num_environments - 1:
        # Vary number of obstacles and sizes
        num_obstacles = random.randint(2, 5)
        min_size = random.uniform(0.5, 1.5)
        max_size = random.uniform(1.5, 3.0)
        
        # Generate random start and goal with minimum distance
        start, goal = generate_random_start_goal(min_distance=5.0)
        
        env = generate_random_environment(
            num_obstacles=num_obstacles,
            min_obstacle_size=min_size,
            max_obstacle_size=max_size,
            start=start,
            goal=goal
        )
        env["id"] = f"env_{i+1}"
        
        # Validate solvability if enabled
        if validate_solvability:
            # Show progress periodically
            if i % 10 == 0:
                print(f"Generated and validating environment {i+1}/{num_environments-1}")
            
            solvable = is_environment_solvable(env, max_validation_iters)
            if not solvable:
                validation_failures += 1
                if validation_failures % 5 == 0:
                    print(f"Warning: Failed validation {validation_failures} times. Still trying...")
                continue  # Skip this environment and try again
        
        # Add valid environment to the collection
        environments.append(env)
        i += 1
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save environments to JSON
    with open(output_file, "w") as f:
        json.dump(environments, f, indent=2)
    
    print(f"Generated {len(environments)} environments and saved to {output_file}")
    if validate_solvability:
        print(f"Solvability validation: rejected {validation_failures} unsolvable environments")
    
    return environments

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    # parser = argparse.ArgumentParser(description="Generate random environment configurations")
    # parser.add_argument("--num", type=int, default=200, help="Number of environments to generate")
    # parser.add_argument("--output", type=str, default="environments/env_configurations.json", 
    #                     help="Output JSON file path")
    # parser.add_argument("--validate", action="store_true", help="Validate environment solvability")
    # parser.add_argument("--max-validation-iters", type=int, default=5000, 
    #                     help="Maximum iterations for validation")
    


    # args = parser.parse_args()

    NUM_ENVIRONMENTS = 200
    ENV_CONFIG_FILE = "environments/env_configurations.json"
    VALIDATE_SOLVABILITY = True
    MAX_VALIDATION_ITERS = 50000
    
    # Generate environments
    generate_environment_set(
        num_environments=NUM_ENVIRONMENTS,
        output_file=ENV_CONFIG_FILE,
        validate_solvability=VALIDATE_SOLVABILITY,
        max_validation_iters=MAX_VALIDATION_ITERS
    ) 
"""
Environment loader for pathfinding algorithm benchmarks.
Loads environment configurations from JSON files.
"""

import json
import os
import ast

def load_environments(config_file="environments/env_configurations.json"):
    """
    Load environments from a JSON configuration file.
    
    Args:
        config_file: Path to JSON file containing environment configurations
        
    Returns:
        List of environment dictionaries
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Environment configuration file {config_file} not found")
    
    with open(config_file, "r") as f:
        data = json.load(f)
    
    # Convert string tuples back to actual tuples
    environments = []
    for env in data:
        # Convert string representations of tuples to actual tuples
        if isinstance(env["start"], str):
            env["start"] = ast.literal_eval(env["start"])
        if isinstance(env["goal"], str):
            env["goal"] = ast.literal_eval(env["goal"])
            
        obstacles = []
        for obstacle in env["obstacles"]:
            if isinstance(obstacle, str):
                obstacle = ast.literal_eval(obstacle)
            obstacles.append(obstacle)
        env["obstacles"] = obstacles
        
        environments.append(env)
    
    return environments

def get_environment_by_id(env_id, config_file="environments/env_configurations.json"):
    """
    Get a specific environment by ID.
    
    Args:
        env_id: Environment ID to load
        config_file: Path to JSON file containing environment configurations
        
    Returns:
        Environment dictionary
    """
    environments = load_environments(config_file)
    
    for env in environments:
        if env["id"] == env_id:
            return env
    
    raise ValueError(f"Environment with ID {env_id} not found in {config_file}")

def get_default_environment():
    """
    Get the default environment.
    
    Returns:
        Default environment dictionary
    """
    return {
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
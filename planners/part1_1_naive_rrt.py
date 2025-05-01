import numpy as np
import matplotlib.pyplot as plt
import random
import time
import json
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    from planners.utils import is_in_collision
except ImportError:
    try:
        from utils import is_in_collision
    except ImportError:
        sys.path.append(current_dir)
        from utils import is_in_collision

def run_naive_rrt(step_size=1.4, max_iters=50000, goal_threshold=0.5, benchmark_mode=False,
                 width=10, height=10, start=(1, 1), goal=(9, 9), obstacles=None, plot_dir=None, plot_number=None):
    """
    Run the naive RRT algorithm with the given parameters.
    
    Args:
        step_size: The step size for extending the tree
        max_iters: Maximum number of iterations
        goal_threshold: Distance threshold to consider goal reached
        benchmark_mode: Whether to suppress plotting and just return results
        width: Width of the environment
        height: Height of the environment
        start: Start position (x, y) - can be a tuple or list
        goal: Goal position (x, y) - can be a tuple or list
        obstacles: List of obstacles as (x, y, width, height)
    
    Returns:
        Dictionary with results of the RRT run
    """

    if obstacles is None:
        obstacles = [
            (3, 3, 2, 1.5),
            (6, 5, 1, 3),
            (2, 7, 2, 1)
        ]
    
    if isinstance(start, list):
        start = tuple(start)
    if isinstance(goal, list):
        goal = tuple(goal)
    
    processed_obstacles = []
    for obs in obstacles:
        if isinstance(obs, list):
            processed_obstacles.append(tuple(obs))
        else:
            processed_obstacles.append(obs)
    obstacles = processed_obstacles

    nodes = [start]
    parents = {start: None}

    def steer(p1, p2, step_size):
        vec = np.array(p2) - np.array(p1)
        length = np.linalg.norm(vec)
        if length == 0:
            return p1
        return tuple(np.array(p1) + step_size * vec / length)

    start_time = time.time()
    goal_reached = False
    goal_iteration = max_iters

    for i in range(max_iters):
        x_tree = random.choice(nodes)
        x_rand = (random.uniform(0, width), random.uniform(0, height))
        x_new = steer(x_tree, x_rand, step_size)

        if not is_in_collision(x_tree, x_new, obstacles):
            nodes.append(x_new)
            parents[x_new] = x_tree

            if np.linalg.norm(np.array(x_new) - np.array(goal)) < goal_threshold:
                parents[goal] = x_new
                nodes.append(goal)
                goal_reached = True
                goal_iteration = i + 1
                break

    end_time = time.time()

    total_cost = None
    if goal_reached:
        total_cost = 0
        node = goal
        while parents[node] is not None:
            total_cost += np.linalg.norm(np.array(node) - np.array(parents[node]))
            node = parents[node]

    result = {
        "success": goal_reached,
        "iterations": goal_iteration if goal_reached else max_iters,
        "path_cost": round(total_cost, 2) if total_cost is not None else None,
        "nodes": len(nodes),
        "runtime": round(end_time - start_time, 4)
    }

    if not benchmark_mode:


        fig, ax = plt.subplots()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_title(
            f"Naive RRT\nStep = {step_size}, Max Iters = {max_iters}, Threshold = {goal_threshold}",
            fontsize=10
        )

        for (ox, oy, ow, oh) in obstacles:
            ax.add_patch(plt.Rectangle((ox, oy), ow, oh, color='gray'))

        for node in nodes:
            parent = parents.get(node)
            if parent:
                ax.plot([node[0], parent[0]], [node[1], parent[1]], 'blue')

        if goal_reached:
            node = goal
            path = []
            while parents[node] is not None:
                path.append((node, parents[node]))
                node = parents[node]
            for p in path:
                ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], 'orange', linewidth=2)

        ax.plot(start[0], start[1], 'go', markersize=8, label='Start')
        ax.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')

        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_dir}/env_{plot_number}.png")
        plt.close()
    
    return result

if __name__ == "__main__":
    default_config_path = os.path.join(os.path.dirname(__file__), "..", "results", "naive_config.json")
    default_result_path = os.path.join(os.path.dirname(__file__), "..", "results", "result.json")

    config_path = os.environ.get("CONFIG_PATH", default_config_path)
    result_path = os.environ.get("RESULT_PATH", default_result_path)

    step_size = 1.4
    max_iters = 50000
    goal_threshold = 0.5
    benchmark_mode = os.environ.get("BENCHMARK_MODE") == "1" 

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        step_size = config.get("step_size", step_size)
        max_iters = config.get("max_iters", max_iters)

    print(f"[INFO] Running with step_size={step_size}, max_iters={max_iters}")

    result = run_naive_rrt(
        step_size=step_size, 
        max_iters=max_iters, 
        goal_threshold=goal_threshold,
        benchmark_mode=benchmark_mode
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f)

    if benchmark_mode:
        fig_name = f"figures/naive/stepsize{step_size:.1f}_maxiterations{max_iters}_treshold{goal_threshold}.png"
        plt.savefig(fig_name)
        plt.close()

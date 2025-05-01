import numpy as np
import matplotlib.pyplot as plt
import random
import math
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

def run_rrt_star_improved(step_size=1.0, max_iters=3000, goal_threshold=1.0, gamma=2.5, 
                         alpha=1.0, beta=1.0, dimension=2, goal_sample_rate=0.1, benchmark_mode=False,
                         width=10, height=10, start=(1, 1), goal=(9, 9), obstacles=None, plot_dir=None, plot_number=None):
    """
    Run the improved RRT* variant algorithm with the given parameters.
    
    Args:
        step_size: The step size for extending the tree
        max_iters: Maximum number of iterations
        goal_threshold: Distance threshold to consider goal reached
        gamma: RRT* specific parameter for radius calculation
        alpha: Weight for distance cost component
        beta: Weight for goal heuristic cost component
        dimension: Dimension of the space
        goal_sample_rate: Probability of sampling the goal directly
        benchmark_mode: Whether to suppress plotting and just return results
        width: Width of the environment
        height: Height of the environment
        start: Start position (x, y) - can be a tuple or list
        goal: Goal position (x, y) - can be a tuple or list
        obstacles: List of obstacles as (x, y, width, height)
    
    Returns:
        Dictionary with results of the RRT* improved variant run
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
    costs = {start: 0}

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def steer(p1, p2, step_size):
        vec = np.array(p2) - np.array(p1)
        dist = np.linalg.norm(vec)
        if dist == 0:
            return p1
        vec = vec / dist
        return tuple(np.array(p1) + step_size * vec)

    def nearest_node(nodes, x_rand):
        return min(nodes, key=lambda n: distance(n, x_rand))

    def near_nodes(nodes, x_new):
        n = len(nodes)
        r = min(gamma * (math.log(n + 1) / (n + 1))**(1/dimension), step_size * 5)
        return [node for node in nodes if distance(node, x_new) <= r]

    start_time = time.time()
    goal_reached = False
    goal_iteration = max_iters

    for i in range(max_iters):
        if random.random() < goal_sample_rate:
            x_rand = goal
        else:
            x_rand = (random.uniform(0, width), random.uniform(0, height))

        x_nearest = nearest_node(nodes, x_rand)
        x_new = steer(x_nearest, x_rand, step_size)

        if is_in_collision(x_nearest, x_new, obstacles):
            continue

        near = near_nodes(nodes, x_new)
        min_parent = x_nearest
        min_cost = costs[x_nearest] + alpha * distance(x_nearest, x_new) + beta * distance(x_rand, goal)

        for node in near:
            if not is_in_collision(node, x_new, obstacles):
                new_cost = costs[node] + alpha * distance(node, x_new) + beta * distance(x_rand, goal)
                if new_cost < min_cost:
                    min_cost = new_cost
                    min_parent = node

        nodes.append(x_new)
        parents[x_new] = min_parent
        costs[x_new] = costs[min_parent] + distance(min_parent, x_new)

        for node in near:
            if node == min_parent:
                continue
            x_rewire = steer(x_new, node, step_size)
            if distance(x_new, node) < step_size + 1e-6 and not is_in_collision(x_new, node, obstacles):
                potential_cost = costs[x_new] + distance(x_new, node)
                if potential_cost < costs[node]:
                    parents[node] = x_new
                    costs[node] = potential_cost

        if distance(x_new, goal) < goal_threshold:
            parents[goal] = x_new
            costs[goal] = costs[x_new] + distance(x_new, goal)
            nodes.append(goal)
            goal_reached = True
            goal_iteration = i + 1
            if not benchmark_mode:
                print(f"Goal reached in {i+1} iterations.")
            break
    else:
        if not benchmark_mode:
            print("Failed to reach goal.")

    end_time = time.time()

    total_cost = None
    if goal_reached:
        total_cost = 0
        node = goal
        while parents[node] is not None:
            total_cost += distance(node, parents[node])
            node = parents[node]

    result = {
        "success": goal_reached,
        "iterations": goal_iteration,
        "path_cost": round(total_cost, 2) if total_cost else None,
        "nodes": len(nodes),
        "runtime": round(end_time - start_time, 4)
    }

    if not benchmark_mode:
        fig, ax = plt.subplots()
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect('equal')
        ax.set_title(
            f"Improved RRT* Variant\nα={alpha}, β={beta}, Goal Bias={goal_sample_rate}, Nodes={len(nodes)}",
            fontsize=10
        )

        for (ox, oy, ow, oh) in obstacles:
            ax.add_patch(plt.Rectangle((ox, oy), ow, oh, color='gray'))

        for node in nodes:
            parent = parents.get(node)
            if parent:
                ax.plot([node[0], parent[0]], [node[1], parent[1]], 'blue')

        # Highlight path to goal if found
        if goal_reached:
            node = goal
            path = []
            while parents[node] is not None:
                path.append((node, parents[node]))
                node = parents[node]
            for p in path:
                # make this orange
                ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], 'orange', linewidth=2)

        ax.plot(start[0], start[1], 'go', markersize=8, label='Start')
        ax.plot(goal[0], goal[1], 'ro', markersize=8, label='Goal')

        plt.legend()
        plt.grid(True)
        plt.savefig(f"{plot_dir}/env_{plot_number}.png")
        plt.close()
    
    return result

if __name__ == "__main__":
    result = run_rrt_star_improved(benchmark_mode=False)
    
    os.makedirs("results", exist_ok=True)
    with open("results/result.json", "w") as f:
        json.dump(result, f)

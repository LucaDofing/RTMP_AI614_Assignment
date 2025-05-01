import json
import pandas as pd
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Add at the top of your script
total_start_time = time.time()

# Add the parent directory to sys.path so we can import planners
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the planner functions from all the modules
from planners.part1_1_naive_rrt import run_naive_rrt
from planners.part1_2_heuristic_rrt import run_heuristic_rrt
from planners.part1_3_rrt import run_standard_rrt
from planners.part2_1_rrt_star import run_rrt_star

# Define available planners with their function references
PLANNERS = {
    "naive_rrt": run_naive_rrt,
    "heuristic_rrt": run_heuristic_rrt,
    "standard_rrt": run_standard_rrt,
    "rrt_star": run_rrt_star
}

# === Parameter grid ===
# Which planners to benchmark (add/remove from this list to include different planners)
planner_types = ["heuristic_rrt"]
step_sizes = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
max_iters = [500]  # 300 is not a limiting factor here (most need 170-240)

# Create full parameter grid
param_grid = list(itertools.product(planner_types, step_sizes, max_iters))

# === How many trials per parameter combination ===
n_trials = 10  # Full benchmark

# === Output setup ===
os.makedirs("results", exist_ok=True)

def run_single_trial(params_with_trial):
    (planner_type, step_size, max_iter, trial_id) = params_with_trial
    
    try:
        # Get the appropriate planner function
        planner_func = PLANNERS.get(planner_type)
        if not planner_func:
            raise ValueError(f"Unknown planner type: {planner_type}")
            
        # Run the RRT algorithm directly, without subprocess or file I/O
        result = planner_func(
            step_size=step_size,
            max_iters=max_iter,
            benchmark_mode=True
        )
        
        # Add parameters to result
        result["planner_type"] = planner_type
        result["step_size"] = step_size
        result["max_iters"] = max_iter
        result["trial_id"] = trial_id

    except Exception as e:
        result = {
            "planner_type": planner_type,
            "step_size": step_size,
            "max_iters": max_iter,
            "trial_id": trial_id,
            "success": False,
            "error": str(e),
            "path_cost": None,
            "iterations": None,
            "runtime": None,
            "nodes": None
        }

    return result

if __name__ == '__main__':
    # === Build trial list ===
    trial_grid = []
    for params in param_grid:
        for trial_id in range(n_trials):
            # Add trial_id to the parameter set
            trial_grid.append((params[0], params[1], params[2], trial_id))

    total_runs = len(trial_grid)
    print(f"\n🔧 Running parameter search with {os.cpu_count()} parallel workers...")
    print(f"🔁 {total_runs} total runs ({n_trials} trials per setting)")
    print(f"📊 Testing planners: {', '.join(planner_types)}\n")

    # === Run all trials in parallel with progress reporting ===
    results = []
    completed = 0
    progress_step = max(1, total_runs // 10)  # Report every 10%
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all tasks and get Future objects
        future_to_params = {executor.submit(run_single_trial, params): params for params in trial_grid}
        
        # Process results as they complete
        for future in as_completed(future_to_params):
            results.append(future.result())
            completed += 1
            
            # Report progress at each 10% milestone
            if completed % progress_step == 0 or completed == total_runs:
                percent_done = (completed / total_runs) * 100
                elapsed_time = time.time() - total_start_time
                est_total_time = elapsed_time / (completed / total_runs)
                est_remaining = est_total_time - elapsed_time
                
                print(f"Progress: {percent_done:.1f}% ({completed}/{total_runs}) - "
                      f"Elapsed: {elapsed_time:.1f}s - "
                      f"Est. remaining: {est_remaining:.1f}s")

    # === Save all raw results ===
    df = pd.DataFrame(results)
    df.to_csv("results/planner_param_trials_raw.csv", index=False)
    print("\n✅ All raw trials saved to results/planner_param_trials_raw.csv")

    # === Aggregate ===
    summary = df.groupby(["planner_type", "step_size", "max_iters"]).agg({
        "success": ["mean"],
        "path_cost": ["mean", "std", "min", "max"],
        "iterations": ["mean", "std", "min", "max"],
        "runtime": ["mean", "std", "min", "max"],
        "nodes": ["mean", "std"]
    }).round(2)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index(inplace=True)

    summary.to_csv("results/planner_param_summary.csv", index=False)
    print("\n✅ Summary saved to results/planner_param_summary.csv")
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    print(f"\n🏁 Total runtime: {total_runtime:.2f} seconds")
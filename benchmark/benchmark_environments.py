"""
Environment benchmark script for pathfinding algorithms.

This script runs various planners across a set of different obstacle configurations
and reports performance statistics.
"""

import json
import pandas as pd
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the planner functions
from planners.part1_1_naive_rrt import run_naive_rrt
from planners.part1_2_heuristic_rrt import run_heuristic_rrt
from planners.part1_3_rrt import run_standard_rrt
from planners.part2_1_rrt_star import run_rrt_star
from planners.part2_2_rrt_star_variant import run_rrt_star_variant
from planners.part2_2_rrt_star_improved import run_rrt_star_improved
from environments.generator import generate_environment_set
from environments.loader import load_environments

# Define available planners with their function references
PLANNERS = {
    "naive_rrt": run_naive_rrt,
    "heuristic_rrt": run_heuristic_rrt,
    "standard_rrt": run_standard_rrt,
    "rrt_star": run_rrt_star,
    "rrt_star_variant": run_rrt_star_variant,
    "rrt_star_improved": run_rrt_star_improved
}


# Number of different environments to use (default: 200)
NUM_ENVIRONMENTS = 200

# Number of trials per configuration (default: 1)
TRIALS_PER_CONFIG = 100

# Which planners to evaluate (uncomment/comment to select)
PLANNERS_TO_TEST = [
    "naive_rrt",
    "heuristic_rrt", 
    "standard_rrt", 
    "rrt_star", 
    "rrt_star_variant",
    "rrt_star_improved"
]

# Step sizes to evaluate
STEP_SIZES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

# Maximum iterations for each planner run
MAX_ITERATIONS = [1000]

# Environment configuration file
ENV_CONFIG_FILE = "environments/env_configurations.json"


def run_single_benchmark(args):
    """
    Run a single benchmark case.
    
    Args:
        args: Tuple of (planner_name, env, step_size, max_iters, trial_id)
        
    Returns:
        Dictionary with benchmark results
    """
    planner_name, env, step_size, max_iters, trial_id = args
    
    try:
        planner_func = PLANNERS.get(planner_name)
        if not planner_func:
            raise ValueError(f"Unknown planner type: {planner_name}")
        
        # Extract environment parameters
        width = env.get("width", 10)
        height = env.get("height", 10)
        start = env.get("start", (1, 1))
        goal = env.get("goal", (9, 9))
        obstacles = env.get("obstacles", [])
        
        # Run the planner
        result = planner_func(
            step_size=step_size,
            max_iters=max_iters,
            benchmark_mode=True,
            width=width,
            height=height,
            start=start,
            goal=goal,
            obstacles=obstacles
        )
        
        # Add metadata to result
        result["planner_type"] = planner_name
        result["env_id"] = env.get("id", "unknown")
        result["step_size"] = step_size
        result["max_iters"] = max_iters
        result["trial_id"] = trial_id
        
    except Exception as e:
        result = {
            "planner_type": planner_name,
            "env_id": env.get("id", "unknown"),
            "step_size": step_size,
            "max_iters": max_iters,
            "trial_id": trial_id,
            "success": False,
            "error": str(e),
            "path_cost": None,
            "iterations": None,
            "runtime": None,
            "nodes": None
        }
    
    return result

def main():
    """Run the benchmark across multiple environments and planners."""
    # Record total benchmark start time
    total_start_time = time.time()
    # delete bigbenchmark directory if it exists including all subdirectories
    if os.path.exists("figures/bigbenchmark"):
        for file in os.listdir("figures/bigbenchmark"):
            file_path = os.path.join("figures/bigbenchmark", file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)
    # Make sure environments are generated
    if not os.path.exists(ENV_CONFIG_FILE):
        print("Generating environment configurations...")
        generate_environment_set(NUM_ENVIRONMENTS, ENV_CONFIG_FILE)
    
    # Load environments
    print("Loading environment configurations...")
    environments = load_environments(ENV_CONFIG_FILE)
    
    # Limit to the specified number of environments
    environments = environments[:NUM_ENVIRONMENTS]
    
    # Create benchmark task list
    task_list = []
    for env in environments:
        for planner_name in PLANNERS_TO_TEST:
            for step_size in STEP_SIZES:
                for max_iters in MAX_ITERATIONS:
                    for trial_id in range(TRIALS_PER_CONFIG):
                        task_list.append((planner_name, env, step_size, max_iters, trial_id))
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Report benchmark setup
    total_runs = len(task_list)
    print(f"\n Running environment benchmark with {os.cpu_count()} parallel workers...")
    print(f"{total_runs} total runs ({TRIALS_PER_CONFIG} trials per setting)")
    print(f"Testing planners: {', '.join(PLANNERS_TO_TEST)}")
    print(f"Using {len(environments)} different environments")
    
    # Run benchmarks in parallel
    results = []
    completed = 0
    # Replace progress_step calculation with last_update_time
    last_update_time = time.time()
    update_interval = 10  # seconds
    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(run_single_benchmark, task): task for task in task_list}
        
        # Process results as they complete
        for future in as_completed(future_to_task):
            results.append(future.result())
            completed += 1
            
            # Report progress every 10 seconds or at completion
            current_time = time.time()
            if (current_time - last_update_time >= update_interval) or completed == total_runs:
                percent_done = (completed / total_runs) * 100
                elapsed_time = current_time - total_start_time
                est_total_time = elapsed_time / (completed / total_runs)
                est_remaining = est_total_time - elapsed_time
                
                print(f"Progress: {percent_done:.1f}% ({completed}/{total_runs}) - "
                      f"Elapsed: {elapsed_time:.1f}s - "
                      f"Est. remaining: {est_remaining:.1f}s")
                      
                # Update the last update time
                last_update_time = current_time
    
    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv("results/environment_benchmark_raw.csv", index=False)
    print("\n All raw results saved to results/environment_benchmark_raw.csv")
    
    # Generate summary statistics
    summary = df.groupby(["planner_type", "env_id", "step_size", "max_iters"]).agg({
        "success": ["mean"],
        "path_cost": ["mean", "std", "min", "max"],
        "iterations": ["mean", "std", "min", "max"],
        "runtime": ["mean", "std", "min", "max"],
        "nodes": ["mean", "std"]
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary.reset_index(inplace=True)
    
    # Save summary
    summary.to_csv("results/environment_benchmark_summary.csv", index=False)
    print("\n Summary saved to results/environment_benchmark_summary.csv")
    
    # Generate planner summary across all environments
    planner_summary = df.groupby(["planner_type", "step_size"]).agg({
        "success": ["mean"],
        "path_cost": ["mean", "std", "min", "max"],
        "iterations": ["mean", "std", "min", "max"],
        "runtime": ["mean", "std", "min", "max"],
        "nodes": ["mean", "std"]
    }).round(2)
    
    planner_summary.columns = ['_'.join(col).strip() for col in planner_summary.columns.values]
    planner_summary.reset_index(inplace=True)
    
    # Save planner summary
    planner_summary.to_csv("results/planner_environment_summary.csv", index=False)
    print("\n Planner summary saved to results/planner_environment_summary.csv")
    
    # Report total runtime
    total_end_time = time.time()
    total_runtime = total_end_time - total_start_time
    print(f"\n Total benchmark runtime: {total_runtime:.2f} seconds")
    
    # Generate visualization plots
    print("\n Generating benchmark visualizations...")
    create_benchmark_visualizations(df, summary, planner_summary, environments)

def create_benchmark_visualizations(raw_data, summary, planner_summary, environments):
    """
    Create visualizations of benchmark results in various formats.
    
    Args:
        raw_data: Pandas DataFrame with all raw benchmark results
        summary: Pandas DataFrame with summary by planner, environment, and step size
        planner_summary: Pandas DataFrame with summary by planner and step size
        environments: List of environment dictionaries used in the benchmark
    """
    try:
        # Get the unique planners that have data in the results
        active_planners = raw_data["planner_type"].unique()
        
        # Create bigbenchmark directory in figures
        bigbenchmark_dir = "figures/bigbenchmark"
        try:
            os.makedirs(bigbenchmark_dir, exist_ok=True)

            # clear the directory if it exists
            if os.path.exists(bigbenchmark_dir):
                for file in os.listdir(bigbenchmark_dir):
                    file_path = os.path.join(bigbenchmark_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
        except PermissionError:
            print(f"⚠️ Permission error creating or cleaning directory: {bigbenchmark_dir}")
            # Continue with the visualization even if we can't clear the directory
        
        # Create subdirectories for each active planner
        for planner in active_planners:
            try:
                os.makedirs(f"{bigbenchmark_dir}/{planner}", exist_ok=True)
            except PermissionError:
                print(f"⚠️ Permission error creating directory: {bigbenchmark_dir}/{planner}")
                # Skip directory creation but continue with other visualizations
        
        # Set global plotting style
        plt.style.use('ggplot')
        
        # === 1. Overall Performance Comparison ===
        
        # Success rate by planner and step size
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=planner_summary, x="step_size", y="success_mean", hue="planner_type", marker='o')
        plt.title("Success Rate by Planner and Step Size")
        plt.xlabel("Step Size")
        plt.ylabel("Success Rate")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{bigbenchmark_dir}/success_rate_comparison.png")
        plt.close()
        
        # Path cost by planner and step size
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=planner_summary, x="step_size", y="path_cost_mean", hue="planner_type", marker='o')
        plt.title("Path Cost by Planner and Step Size")
        plt.xlabel("Step Size")
        plt.ylabel("Path Cost")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{bigbenchmark_dir}/path_cost_comparison.png")
        plt.close()
        
        # Runtime by planner and step size
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=planner_summary, x="step_size", y="runtime_mean", hue="planner_type", marker='o')
        plt.title("Runtime by Planner and Step Size")
        plt.xlabel("Step Size")
        plt.ylabel("Runtime (seconds)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{bigbenchmark_dir}/runtime_comparison.png")
        plt.close()
        
        # Node count by planner and step size
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=planner_summary, x="step_size", y="nodes_mean", hue="planner_type", marker='o')
        plt.title("Node Count by Planner and Step Size")
        plt.xlabel("Step Size")
        plt.ylabel("Number of Nodes")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{bigbenchmark_dir}/nodes_comparison.png")
        plt.close()
        
        # === 2. Success-Runtime Tradeoff Scatter Plot ===
        plt.figure(figsize=(12, 8))
        scatter_data = planner_summary[planner_summary["success_mean"] > 0].copy()
        
        for planner in PLANNERS_TO_TEST:
            planner_data = scatter_data[scatter_data["planner_type"] == planner]
            plt.scatter(planner_data["runtime_mean"], planner_data["path_cost_mean"], 
                       s=planner_data["success_mean"]*100, alpha=0.7, label=planner)
        
        plt.title("Success-Runtime-Path Cost Tradeoff")
        plt.xlabel("Runtime (seconds)")
        plt.ylabel("Path Cost")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{bigbenchmark_dir}/tradeoff_analysis.png")
        plt.close()
        
        # === 3. Box Plots for Comparing Distributions ===
        metrics = ["path_cost", "runtime", "iterations", "nodes"]
        
        for metric in metrics:
            plt.figure(figsize=(14, 8))
            boxplot_data = raw_data[raw_data["success"] == True].copy()  # Only successful runs
            sns.boxplot(data=boxplot_data, x="planner_type", y=metric)
            plt.title(f"Distribution of {metric} by Planner Type")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{bigbenchmark_dir}/{metric}_distribution.png")
            plt.close()
        
        # === 4. Step Size Impact Analysis ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Success rate
        for planner in PLANNERS_TO_TEST:
            planner_data = planner_summary[planner_summary["planner_type"] == planner]
            axes[0, 0].plot(planner_data["step_size"], planner_data["success_mean"], marker='o', label=planner)
        axes[0, 0].set_title("Success Rate")
        axes[0, 0].set_ylabel("Success Rate")
        axes[0, 0].set_ylim(0, 1.05)
        axes[0, 0].grid(True)
        
        # Path cost
        for planner in PLANNERS_TO_TEST:
            planner_data = planner_summary[planner_summary["planner_type"] == planner]
            axes[0, 1].plot(planner_data["step_size"], planner_data["path_cost_mean"], marker='o', label=planner)
        axes[0, 1].set_title("Path Cost")
        axes[0, 1].set_ylabel("Path Cost")
        axes[0, 1].grid(True)
        
        # Runtime
        for planner in PLANNERS_TO_TEST:
            planner_data = planner_summary[planner_summary["planner_type"] == planner]
            axes[1, 0].plot(planner_data["step_size"], planner_data["runtime_mean"], marker='o', label=planner)
        axes[1, 0].set_title("Runtime")
        axes[1, 0].set_ylabel("Seconds")
        axes[1, 0].set_xlabel("Step Size")
        axes[1, 0].grid(True)
        
        # Node count
        for planner in PLANNERS_TO_TEST:
            planner_data = planner_summary[planner_summary["planner_type"] == planner]
            axes[1, 1].plot(planner_data["step_size"], planner_data["nodes_mean"], marker='o', label=planner)
        axes[1, 1].set_title("Node Count")
        axes[1, 1].set_ylabel("Nodes")
        axes[1, 1].set_xlabel("Step Size")
        axes[1, 1].grid(True)
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(PLANNERS_TO_TEST))
        plt.tight_layout()
        plt.savefig(f"{bigbenchmark_dir}/step_size_impact.png")
        plt.close()
        
        # === 5. Individual Planner Performance Across 10 Different Environments ===
        for planner in PLANNERS_TO_TEST:
            # Skip planners that don't have data
            if planner not in active_planners:
                continue
                
            # Get subset of data for this planner
            planner_data = raw_data[raw_data["planner_type"] == planner]
            
            # Skip if no data for this planner
            if len(planner_data) == 0:
                continue
            
            # Group by environment
            env_groups = planner_data.groupby("env_id")
            
            # Select 10 different environments with different obstacle configurations
            env_sample = list(env_groups.groups.keys())[:10]  # Take first 10 environments
            
            # Plot for each environment
            for i, env_id in enumerate(env_sample):
                # run the planner on the env with benchmark_mode turned off
                planner_func = PLANNERS.get(planner)
                
                # Extract environment data for this env_id
                env_data = next((env for env in environments if env["id"] == env_id), None)
                if env_data:
                    # Create plot directory for this planner if it doesn't exist
                    planner_plot_dir = f"{bigbenchmark_dir}/{planner}"
                    try:
                        os.makedirs(planner_plot_dir, exist_ok=True)
                        
                        # Debug the obstacle data
                        obstacles = env_data.get("obstacles", [])
                        #print(f"Environment {env_id} has {len(obstacles)} obstacles: {obstacles[:2]}...")
                        
                        # Run the planner with benchmark_mode=False to generate plot
                        planner_func(
                            step_size=1.0,  # Use a reasonable step size
                            max_iters=1000,
                            benchmark_mode=False,
                            width=env_data.get("width", 10),
                            height=env_data.get("height", 10),
                            start=env_data.get("start", (1, 1)),
                            goal=env_data.get("goal", (9, 9)),
                            obstacles=obstacles,
                            plot_dir=planner_plot_dir,
                            plot_number=env_id
                        )
                    except PermissionError as e:
                        print(f"⚠️ Permission error creating directory for {planner}: {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Error running planner {planner} on environment {env_id}: {e}")
                        continue

        # === 6. Correlation Analysis ===
        # Calculate correlations between metrics for each planner
        plt.figure(figsize=(12, 10))
        
        for i, planner in enumerate(PLANNERS_TO_TEST):
            planner_data = raw_data[raw_data["planner_type"] == planner]
            
            # Select only numeric columns
            numeric_data = planner_data.select_dtypes(include=[np.number])
            
            # Remove columns that don't make sense for correlation
            for col in ["trial_id", "max_iters"]:
                if col in numeric_data.columns:
                    numeric_data = numeric_data.drop(columns=[col])
            
            # Calculate correlation matrix
            corr = numeric_data.corr()
            
            # Plot correlation matrix
            plt.subplot(2, 3, i+1)
            sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                       square=True, linewidths=.5, fmt=".2f", cbar=False)
            plt.title(f"{planner} Metric Correlations")
            plt.tight_layout()
        
        plt.savefig(f"{bigbenchmark_dir}/metric_correlations.png")
        plt.close()
        
        # === 7. Efficiency Analysis: Path Cost / Runtime ratio ===
        # Higher values mean more efficient algorithms (better path at lower computational cost)
        plt.figure(figsize=(12, 8))
        
        # For each planner and step size, calculate efficiency as path_cost / runtime
        efficiency_data = planner_summary.copy()
        efficiency_data["efficiency"] = efficiency_data["path_cost_mean"] / efficiency_data["runtime_mean"]
        
        # Invert if lower path costs are better (this depends on what path_cost represents)
        # efficiency_data["efficiency"] = 1 / efficiency_data["efficiency"]
        
        sns.lineplot(data=efficiency_data, x="step_size", y="efficiency", hue="planner_type", marker='o')
        plt.title("Algorithm Efficiency (Path Quality per Computation Time)")
        plt.xlabel("Step Size")
        plt.ylabel("Efficiency Ratio")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{bigbenchmark_dir}/efficiency_ratio.png")
        plt.close()
        
        print(f"\n All benchmark visualizations saved to {bigbenchmark_dir}/")
        print("  - Overall performance comparisons")
        print("  - Success-Runtime-Path Cost tradeoff analysis")
        print("  - Metric distributions")
        print("  - Step size impact analysis")
        print("  - Individual planner performance on 10 different environments")
        print("  - Metric correlation analysis")
        print("  - Efficiency ratio analysis")
        
    except Exception as e:
        print(f"\n⚠️ Could not generate visualizations: {e}")
        print("Please install matplotlib and seaborn: pip install matplotlib seaborn")

if __name__ == "__main__":
    main() 
# RRT Pathfinding Benchmark

This repository contains the implementations and a benchmark suite for Rapidly-exploring Random Trees (RRT) and its variants for pathfinding in 2D environments with obstacles.

[GitHub Repository](https://github.com/LucaDofing/RTMP_AI614_Assignment)

## Overview

The project implements and evaluates multiple RRT-based algorithms on various randomly generated environments. The benchmark measures performance metrics such as success rate, path cost, runtime, and node expansion across different parameter configurations.

## Algorithms Implemented

The following RRT variants are implemented and benchmarked:

1. **Naive RRT** (`naive_rrt`): Basic implementation without goal biasing or optimizations
2. **Heuristic RRT** (`heuristic_rrt`): RRT with goal-biasing heuristic
3. **Standard RRT** (`standard_rrt`): Conventional RRT implementation
4. **RRT*** (`rrt_star`): RRT with asymptotic optimality guarantees
5. **RRT*** **Variant** (`rrt_star_variant`): Modified version of RRT*
6. **RRT*** **Improved** (`rrt_star_improved`): Enhanced RRT* with optimizations for faster convergence and higher efficiency

## Repository Structure

- **planners/**: Contains implementations of all RRT variants
  - `part0_environment.py`: Base environment representation
  - `part1_1_naive_rrt.py`: Implementation of Naive RRT
  - `part1_2_heuristic_rrt.py`: Implementation of Heuristic RRT
  - `part1_3_rrt.py`: Implementation of Standard RRT
  - `part2_1_rrt_star.py`: Implementation of RRT*
  - `part2_2_rrt_star_variant.py`: Implementation of RRT* Variant
  - `part2_2_rrt_star_improved.py`: Implementation of optimized RRT*
  - `utils.py`: Utility functions for all planners

- **environments/**: Environment generation and management
  - `generator.py`: Creates random obstacle configurations
  - `loader.py`: Loads environment configurations
  - `env_configurations.json`: Stored environment configurations
  - `visualize.py`: Environment visualization utilities
  - `analyze.py`: Environment analysis tools

- **benchmark/**: Benchmarking infrastructure
  - `benchmark_environments.py`: Main benchmarking script
  - `plot_planner_results.py`: Visualization for planner results
  - `plot_heuristic_param_summary.py`: Heuristic parameter analysis
  - `search_heuristic_params_parallel.py`: Parameter search utilities

- **figures/**: Contains generated visualizations
  - `bigbenchmark/`: Comprehensive benchmark visualizations

- **results/**: Benchmark results
  - `environment_benchmark_raw.csv`: Raw benchmark data (1.32 million runs)
  - `environment_benchmark_summary.csv`: Summarized results by environment
  - `planner_environment_summary.csv`: Summarized results by planner and parameter

## Benchmark Setup

The benchmark evaluates each planner across:
- 200 different randomly generated environments
- 11 different step sizes (0.5 to 1.5)
- 100 trials per configuration
- Fixed maximum iterations (1000)

This results in a comprehensive benchmark of 1.32 million total algorithm runs, providing robust statistical analysis of algorithm performance.

## Results

The benchmark measures and compares:
- **Success Rate**: Percentage of runs that successfully find a path
- **Path Cost**: Length of the found path (lower is better)
- **Runtime**: Computational time required (in seconds)
- **Node Count**: Number of nodes expanded (efficiency metric)
- **Efficiency Ratio**: Path quality per computation time

Visualizations in the `figures/bigbenchmark/` directory provide detailed comparisons across algorithms and parameters.


## Usage

To run the benchmark:

```bash
python benchmark/benchmark_environments.py
```


## Dependencies

- NumPy
- Matplotlib
- Pandas
- Seaborn

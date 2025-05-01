import pandas as pd
import matplotlib.pyplot as plt

# Load summary
summary = pd.read_csv("results/real_planner_summary.csv")

# Set global style
plt.style.use("ggplot")
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

# --- 1. Success Rate ---
plt.figure(figsize=(8, 5))
plt.bar(summary["method"], summary["success_mean"], color=colors)
plt.ylim(0, 1.1)
plt.title("Success Rate by Method")
plt.ylabel("Success Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/plot_success_rate.png")
plt.close()

# --- 2. Iterations ---
plt.figure(figsize=(8, 5))
plt.bar(summary["method"], summary["iterations_mean"], color=colors)
plt.title("Mean Iterations to Goal")
plt.ylabel("Iterations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/plot_iterations.png")
plt.close()

# --- 3. Path Cost ---
plt.figure(figsize=(8, 5))
plt.bar(summary["method"], summary["path_cost_mean"], color=colors)
plt.title("Mean Path Cost")
plt.ylabel("Path Cost")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/plot_path_cost.png")
plt.close()

# --- 4. Runtime ---
plt.figure(figsize=(8, 5))
plt.bar(summary["method"], summary["runtime_mean"], color=colors)
plt.title("Mean Runtime (s)")
plt.ylabel("Seconds")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("figures/plot_runtime.png")
plt.close()

print("Done - Plots saved:")
print("- plot_success_rate.png")
print("- plot_iterations.png")
print("- plot_path_cost.png")
print("- plot_runtime.png")

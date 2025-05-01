import pandas as pd
import matplotlib.pyplot as plt

# Load the summary CSV (from your search script)
summary = pd.read_csv("results/heuristic_param_summary.csv")

# Sort for nicer plots
summary = summary.sort_values(by="step_size")

# Create figures directory if not exists
import os
os.makedirs("figures", exist_ok=True)

# === 1. Success Rate Plot ===
plt.figure(figsize=(10, 6))
plt.plot(summary["step_size"], summary["success_mean"], marker="o")
plt.title("Success Rate vs Step Size")
plt.xlabel("Step Size")
plt.ylabel("Success Rate")
plt.ylim(0, 1.05)
plt.grid(True)
plt.savefig("figures/heuristic_success_rate.png")
plt.close()

# === 2. Path Cost with Std Error Bars ===
plt.figure(figsize=(10, 6))
plt.errorbar(summary["step_size"], summary["path_cost_mean"], 
             yerr=summary["path_cost_std"], fmt='-o', capsize=5)
plt.title("Mean Path Cost vs Step Size")
plt.xlabel("Step Size")
plt.ylabel("Path Cost")
plt.grid(True)
plt.savefig("figures/heuristic_path_cost.png")
plt.close()

# === 3. Runtime with Std Error Bars ===
plt.figure(figsize=(10, 6))
plt.errorbar(summary["step_size"], summary["runtime_mean"], 
             yerr=summary["runtime_std"], fmt='-o', capsize=5)
plt.title("Mean Runtime vs Step Size")
plt.xlabel("Step Size")
plt.ylabel("Runtime (seconds)")
plt.grid(True)
plt.savefig("figures/heuristic_runtime.png")
plt.close()

print("\n✅ Plots saved to figures/ folder:")
print("- heuristic_success_rate.png")
print("- heuristic_path_cost.png")
print("- heuristic_runtime.png")

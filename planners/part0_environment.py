import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from utils import is_in_collision


# Define the environment size
width, height = 10, 10

# Define start and goal
start = (1, 1)
goal = (9, 9)

# Define obstacles as list of rectangles: (x, y, width, height)
obstacles = [
    (3, 3, 2, 1.5),
    (6, 5, 1, 3),
    (2, 7, 2, 1)
]

# Plot the environment
fig, ax = plt.subplots()
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_aspect('equal')
ax.set_title("TAMP Assignment 1")

# Draw obstacles
for (ox, oy, ow, oh) in obstacles:
    ax.add_patch(Rectangle((ox, oy), ow, oh, color='gray'))

# Draw start and goal
ax.plot(start[0], start[1], 'go', markersize=10, label="Start")
ax.plot(goal[0], goal[1], 'ro', markersize=10, label="Goal")

print("Collision from start to goal:", is_in_collision(start, goal, obstacles))

plt.legend()
plt.grid(True)
plt.show()

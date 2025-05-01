import numpy as np

def is_in_collision(p1, p2, obstacles, resolution=0.01):
    """
    Check if the line between p1 and p2 collides with any obstacle.
    Obstacles are given as (x, y, w, h).
    """
    x_vals = np.linspace(p1[0], p2[0], int(np.linalg.norm(np.subtract(p2, p1)) / resolution))
    y_vals = np.linspace(p1[1], p2[1], int(np.linalg.norm(np.subtract(p2, p1)) / resolution))

    for (ox, oy, ow, oh) in obstacles:
        for x, y in zip(x_vals, y_vals):
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return True  # Collision detected

    return False  # No collision

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque

# ==============================
# SETTINGS
# ==============================
N = 40
OBSTACLE_DENSITY = 0.075
ROBOT_SIZE = 1   # Try 2 or 3, robot is square of this width

start = (2, 2)
goal = (N - 3, N - 3)

# ==============================
# GRID
# ==============================
grid = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if random.random() < OBSTACLE_DENSITY:
            grid[i, j] = 1

# CLEAR start & goal area (avoid obstacle near start/goal)
for dx in range(-2, 3):
    for dy in range(-2, 3):
        if 0 <= start[0]+dx < N and 0 <= start[1]+dy < N:
            grid[start[0]+dx, start[1]+dy] = 0
        if 0 <= goal[0]+dx < N and 0 <= goal[1]+dy < N:
            grid[goal[0]+dx, goal[1]+dy] = 0

# ==============================
# INFLATE OBSTACLES for robot size (to avoid collisions)
# ==============================
def inflate_grid(grid, size):
    inflated = np.zeros_like(grid)
    pad = size - 1
    for x in range(N):
        for y in range(N):
            if grid[x, y] == 1:
                x_min = max(0, x - pad)
                x_max = min(N, x + pad + 1)
                y_min = max(0, y - pad)
                y_max = min(N, y + pad + 1)
                inflated[x_min:x_max, y_min:y_max] = 1
    return inflated

inflated_grid = inflate_grid(grid, ROBOT_SIZE)

# ==============================
# PATHFINDING (BFS) with inflated grid
# ==============================
def is_free(grid, x, y):
    # Check robot footprint is free in inflated grid
    for i in range(ROBOT_SIZE):
        for j in range(ROBOT_SIZE):
            nx, ny = x + i, y + j
            if nx >= N or ny >= N:
                return False
            if grid[nx, ny] == 1:
                return False
    return True

def find_path(grid, start, goal):
    queue = deque([start])
    visited = set([start])
    parent = {}

    while queue:
        x, y = queue.popleft()

        if (x, y) == goal:
            break

        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < N and 0 <= ny < N:
                if (nx, ny) not in visited and is_free(grid, nx, ny):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (x, y)

    if goal not in parent:
        print("No valid path found!")
        return [start]

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse()
    return path

path = find_path(inflated_grid, start, goal)

# ==============================
# FORKLIFT (RECTANGLE) CLASS
# ==============================
class Forklift:
    def __init__(self):
        self.pos = [random.randint(5, N-5), random.randint(5, N-5)]
        self.dir = random.choice([[1,0],[-1,0],[0,1],[0,-1]])

    def move(self):
        if random.random() < 0.3:
            self.dir = random.choice([[1,0],[-1,0],[0,1],[0,-1]])

        nx = self.pos[0] + self.dir[0]
        ny = self.pos[1] + self.dir[1]

        if 2 < nx < N-2 and 2 < ny < N-2:
            self.pos = [nx, ny]

    def get_cells(self):
        # forklift size 2x3 rectangle footprint
        x, y = self.pos
        cells = []
        for i in range(2):
            for j in range(3):
                cx, cy = x + i, y + j
                if 0 <= cx < N and 0 <= cy < N:
                    cells.append((cx, cy))
        return cells

forklifts = [Forklift() for _ in range(2)]

# ==============================
# ANIMATION
# ==============================
fig, ax = plt.subplots()
step = 0

def forklift_cells():
    cells = set()
    for f in forklifts:
        cells.update(f.get_cells())
    return cells

def inflate_forklift_cells(cells, size):
    temp_grid = np.zeros((N, N))
    pad = size - 1
    for (x, y) in cells:
        x_min = max(0, x - pad)
        x_max = min(N, x + pad + 1)
        y_min = max(0, y - pad)
        y_max = min(N, y + pad + 1)
        temp_grid[x_min:x_max, y_min:y_max] = 1
    return temp_grid

def update(frame):
    global step, path

    ax.clear()

    # Move forklifts
    for f in forklifts:
        f.move()

    # Inflate forklift positions for collision detection
    fl_cells = forklift_cells()
    fl_inflated = inflate_forklift_cells(fl_cells, ROBOT_SIZE)

    # Combine static and forklift obstacles
    combined_grid = np.maximum(inflated_grid, fl_inflated)

    # Check if next step blocked by forklift or static obstacles
    if step < len(path):
        next_pos = path[step]
        if combined_grid[next_pos] == 1:
            # Replan from current position
            current_pos = path[step-1] if step > 0 else start
            new_path = find_path(combined_grid, current_pos, goal)
            if new_path != [current_pos]:
                path = new_path
                step = 0
            else:
                print("No path available, robot stops.")
                # Stop animation if desired: plt.close()
                return

    # Draw obstacles (brown)
    ox, oy = np.where(grid == 1)
    ax.scatter(oy, ox, c='#8B4513', s=100, marker='s')

    # Draw forklifts (orange rectangles)
    for f in forklifts:
        for (x, y) in f.get_cells():
            ax.scatter(y, x, c='orange', s=120, marker='s')

    # Draw path squares light blue behind obstacles
    if len(path) > 1:
        for (x, y) in path:
            ax.scatter(y, x, c='lightblue', s=80, marker='s', alpha=0.6)

    # Draw start and goal as circles (no boxes)
    ax.scatter(start[1], start[0], c='red', s=200)
    ax.scatter(goal[1], goal[0], c='green', s=200)

    # Draw robot as grey square ROBOT_SIZE x ROBOT_SIZE at current path step
    if step < len(path):
        x, y = path[step]
        for i in range(ROBOT_SIZE):
            for j in range(ROBOT_SIZE):
                ax.scatter(y + j, x + i, c='gray', s=140, marker='s')
        step += 1

    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.set_title("Robot Path Planning")

ani = animation.FuncAnimation(fig, update, frames=1000, interval=150)

plt.show()
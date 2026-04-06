import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque

# ==============================
# SETTINGS
# ==============================
N = 40
ROBOT_SIZE = 1   # Robot is square
GRID_SPACING = 1 # Space between obstacles (1 = 1 empty cell, 2 = 2 empty cells, etc.)
OBSTACLE_DENSITY = 0.2
start = (2, 2)
goal = (N - 3, N - 3)

# ==============================
# GRID (true grid, no tiny gaps)
# ==============================
CELL_SIZE = 2  # each "cell" is 2x2 in actual grid
VIRTUAL_N = N // CELL_SIZE  # virtual grid size
grid = np.zeros((N, N))

# Place obstacles in virtual cells
for i in range(VIRTUAL_N):
    for j in range(VIRTUAL_N):
        if random.random() < OBSTACLE_DENSITY:
            # Fill the corresponding actual grid block completely
            for dx in range(CELL_SIZE):
                for dy in range(CELL_SIZE):
                    grid[i*CELL_SIZE + dx, j*CELL_SIZE + dy] = 1

# Clear start & goal area (in actual grid)
for dx in range(-2, 3):
    for dy in range(-2, 3):
        if 0 <= start[0]+dx < N and 0 <= start[1]+dy < N:
            grid[start[0]+dx, start[1]+dy] = 0
        if 0 <= goal[0]+dx < N and 0 <= goal[1]+dy < N:
            grid[goal[0]+dx, goal[1]+dy] = 0

# ==============================
# INFLATE OBSTACLES for robot size
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
# PATHFINDING (BFS)
# ==============================
def is_free(grid, x, y):
    for i in range(ROBOT_SIZE):
        for j in range(ROBOT_SIZE):
            nx, ny = x + i, y + j
            if nx >= N or ny >= N or grid[nx, ny] == 1:
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
# FORKLIFT CLASS
# ==============================
class Forklift:
    def __init__(self):
        self.pos = [random.randint(5, N-5), random.randint(5, N-5)]
        self.dir = random.choice([[1,0],[-1,0],[0,1],[0,-1]])
        self.width = 1
        self.height = 1

    def get_cells(self):
        x, y = self.pos
        cells = []
        for i in range(self.width):
            for j in range(self.height):
                cx, cy = x + i, y + j
                if 0 <= cx < N and 0 <= cy < N:
                    cells.append((cx, cy))
        return cells

    def move(self, robot_cells, grid):
        # Pick a new random direction occasionally
        if random.random() < 0.3:
            self.dir = random.choice([[1,0],[-1,0],[0,1],[0,-1]])

        # Calculate next position
        nx = self.pos[0] + self.dir[0]
        ny = self.pos[1] + self.dir[1]

        # Check if next position is inside grid
        new_cells = [(nx + i, ny + j) for i in range(self.width) for j in range(self.height)]

        # Avoid robot and obstacles
        collision = any(
            (cx < 0 or cx >= N or cy < 0 or cy >= N or grid[cx, cy] == 1 or (cx, cy) in robot_cells)
            for (cx, cy) in new_cells
        )

        if collision:
            # Move in opposite direction to avoid robot/obstacles
            self.dir = [-self.dir[0], -self.dir[1]]
            nx = self.pos[0] + self.dir[0]
            ny = self.pos[1] + self.dir[1]
            # Check again
            new_cells = [(nx + i, ny + j) for i in range(self.width) for j in range(self.height)]
            if not any(
                (cx < 0 or cx >= N or cy < 0 or cy >= N or grid[cx, cy] == 1 or (cx, cy) in robot_cells)
                for (cx, cy) in new_cells
            ):
                self.pos = [nx, ny]
        else:
            self.pos = [nx, ny]

forklifts = [Forklift() for _ in range(5)]

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

# Initialize robot position
robot_pos = start
step = 0
forward = True
full_path = find_path(inflated_grid, start, goal)
dynamic_path = full_path.copy()

def update(frame):
    global step, dynamic_path, forward, full_path, robot_pos
    ax.clear()


    robot_cells = []
    if step < len(path):
        x, y = path[step]
        for i in range(ROBOT_SIZE):
            for j in range(ROBOT_SIZE):
                robot_cells.append((x + i, y + j))

    # Move forklifts safely
    for f in forklifts:
        f.move(robot_cells, inflated_grid)

    fl_cells = forklift_cells()
    fl_inflated = inflate_forklift_cells(fl_cells, ROBOT_SIZE)
    combined_grid = np.maximum(inflated_grid, fl_inflated)

    target = goal if forward else start

    # Replan if next step blocked
    if step < len(dynamic_path):
        next_pos = dynamic_path[step]
        if combined_grid[next_pos] == 1:
            # Replan from robot's current position
            new_path = find_path(combined_grid, robot_pos, target)
            if new_path != [robot_pos]:
                dynamic_path = new_path
                step = 0
            else:
                return  # completely blocked

    # Draw obstacles
    ox, oy = np.where(grid == 1)
    ax.scatter(oy, ox, c='#8B4513', s=100, marker='s')

    # Draw forklifts
    for f in forklifts:
        for (x, y) in f.get_cells():
            ax.scatter(y, x, c='orange', s=120, marker='s')

    # Draw path
    for (x, y) in dynamic_path:
        ax.scatter(y, x, c='lightblue', s=80, marker='s', alpha=0.6)

    # Start & Goal
    ax.scatter(start[1], start[0], c='red', s=200)
    ax.scatter(goal[1], goal[0], c='green', s=200)

    # Draw robot at its current position
    x, y = robot_pos
    for i in range(ROBOT_SIZE):
        for j in range(ROBOT_SIZE):
            ax.scatter(y + j, x + i, c='gray', s=140, marker='s')

    # Move robot along path
    if step < len(dynamic_path):
        robot_pos = dynamic_path[step]  # update current position
        step += 1
        # Reverse if end reached
        if step >= len(dynamic_path):
            forward = not forward
            dynamic_path = full_path if forward else full_path[::-1]
            step = 0

    ax.set_xlim(0, N)
    ax.set_ylim(N, 0)
    ax.set_title("Robot Path Planning - Grid Obstacles")
    
ani = animation.FuncAnimation(fig, update, frames=1000, interval=150)
plt.show()
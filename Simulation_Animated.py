import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
R = 20.0
v = 1.0
N = 100
minRadius = 0.5
maxRadius = 0.5
centerClearance = 2.0
num_runs = 100
dt = 0.1

# generateCoins and segment_circle_intersection_times functions
def generateCoins(N, R, min_radius, max_radius, center_clearance):
    coins = []
    while len(coins) < N:
        r = np.random.uniform(min_radius, max_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        max_r = R - r
        min_r = center_clearance + r
        if max_r <= min_r:
            continue
        dist = np.sqrt(np.random.uniform(min_r**2, max_r**2))
        x = dist * np.cos(theta)
        y = dist * np.sin(theta)
        overlap = False
        for ox, oy, oradius in coins:
            if np.sqrt((x - ox)**2 + (y - oy)**2) < (r + oradius):
                overlap = True
                break
        if not overlap:
            coins.append((x, y, r))
    return coins

def segment_circle_intersection_times(x1, y1, x2, y2, cx, cy, r):
    dx = x2 - x1
    dy = y2 - y1
    a = dx**2 + dy**2
    b = 2 * (x1 * dx + y1 * dy - cx * dx - cy * dy)
    c = x1**2 + y1**2 + cx**2 + cy**2 - 2*(x1*cx + y1*cy) - r**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []
    sqrt_d = np.sqrt(discriminant)
    t1 = (-b - sqrt_d) / (2*a)
    t2 = (-b + sqrt_d) / (2*a)
    return [t for t in [t1, t2] if 0 < t < 1]

# Initialize simulation variables
current_x = 0.0
current_y = 0.0
theta = np.random.uniform(0, 2 * np.pi)
total_time = 0.0
path_x = [current_x]
path_y = [current_y]
run_number = 0
first_coin_hit_times = []
wall_hit_times = []
coin_hit = False
wall_hit = False
coins = generateCoins(N, R, minRadius, maxRadius, centerClearance)
coin_patches = []

# Figure setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-R, R)
ax.set_ylim(-R, R)
ax.set_aspect('equal', 'box')
ax.set_title("Ladybird Simulation")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
outer_circle = plt.Circle((0, 0), R, fill=False, linestyle='--', color='blue', label='Wall')
ax.add_patch(outer_circle)
for (ox, oy, oradius) in coins:
    new_coin = plt.Circle((ox, oy), oradius, color='gold', alpha=0.5)
    ax.add_patch(new_coin)
    coin_patches.append(new_coin)
line, = ax.plot(path_x, path_y, 'r-', lw=2, label='Path')
text = ax.text(-R * 0.9, R * 0.7, "", fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
ax.legend(loc='upper right')

def update(frame):
    global current_x, current_y, theta, total_time, path_x, path_y, run_number
    global first_coin_hit_times, wall_hit_times, coins, coin_patches, coin_hit, wall_hit

    if run_number < num_runs:
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        new_x = current_x + dx
        new_y = current_y + dy
        
        t_min = 1.0
        hit_type = None
        hit_obj = None
        
        t_wall = segment_circle_intersection_times(current_x, current_y, new_x, new_y, 0, 0, R)
        if t_wall:
            t_min_wall = min(t_wall)
            if t_min_wall < t_min:
                t_min = t_min_wall
                hit_type = 'wall'
        
        for idx, (cx, cy, r) in enumerate(coins):
            t_coin = segment_circle_intersection_times(current_x, current_y, new_x, new_y, cx, cy, r)
            if t_coin:
                t_min_coin = min(t_coin)
                if t_min_coin < t_min:
                    t_min = t_min_coin
                    hit_type = 'coin'
                    hit_obj = idx
        
        if hit_type:
            move_t = t_min
            collision_x = current_x + move_t * dx
            collision_y = current_y + move_t * dy
            total_time += move_t * dt
            
            if hit_type == 'coin':
                # Record first coin hit if not already hit
                if not coin_hit:
                    first_coin_hit_times.append(total_time)
                    coin_hit = True
                # Always bounce off coin, regardless of first or subsequent hit
                cx, cy, _ = coins[hit_obj]
                Nx = collision_x - cx
                Ny = collision_y - cy
                norm = np.sqrt(Nx**2 + Ny**2)
                Nx /= norm
                Ny /= norm
                phi = np.arctan2(Ny, Nx)
                delta = np.random.uniform(-np.pi/2, np.pi/2)
                theta = phi + delta
                epsilon = 1e-5
                current_x = collision_x + epsilon * np.cos(theta)
                current_y = collision_y + epsilon * np.sin(theta)
            
            elif hit_type == 'wall':
                # Record first wall hit if not already hit
                if not wall_hit:
                    wall_hit_times.append(total_time)
                    wall_hit = True
                # Always bounce off wall
                Nx = -collision_x
                Ny = -collision_y
                norm = np.sqrt(Nx**2 + Ny**2)
                Nx /= norm
                Ny /= norm
                phi = np.arctan2(Ny, Nx)
                delta = np.random.uniform(-np.pi/4, np.pi/4)
                theta = phi + delta
                epsilon = 1e-5
                current_x = collision_x + epsilon * np.cos(theta)
                current_y = collision_y + epsilon * np.sin(theta)
            
            # Check if both conditions are met to end the run
            if coin_hit and wall_hit:
                run_number += 1
                current_x, current_y = 0.0, 0.0
                theta = np.random.uniform(0, 2 * np.pi)
                total_time = 0.0
                path_x, path_y = [0.0], [0.0]
                coin_hit = False
                wall_hit = False
                for patch in coin_patches:
                    patch.remove()
                coin_patches.clear()
                coins = generateCoins(N, R, minRadius, maxRadius, centerClearance)
                for (ox, oy, oradius) in coins:
                    new_coin = plt.Circle((ox, oy), oradius, color='gold', alpha=0.5)
                    ax.add_patch(new_coin)
                    coin_patches.append(new_coin)
        else:
            current_x = new_x
            current_y = new_y
            total_time += dt
        
        path_x.append(current_x)
        path_y.append(current_y)
        line.set_data(path_x, path_y)
        
        avg_coin_time = np.mean(first_coin_hit_times) if first_coin_hit_times else 0
        avg_wall_time = np.mean(wall_hit_times) if wall_hit_times else 0
        text.set_text(f"Run: {run_number}/{num_runs}\nAvg Coin Time: {avg_coin_time:.2f}\nAvg Wall Time: {avg_wall_time:.2f}")
    
    return [line, text] + coin_patches

ani = animation.FuncAnimation(fig, update, frames=10000, interval=0.0000001, blit=True)
plt.show()
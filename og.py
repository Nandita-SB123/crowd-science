import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.path import Path
import matplotlib.patches as patches

# --- CONFIGURATION ---
NUM_PEOPLE = 150
SIM_SPEED = 1.0
NEIGHBOR_DIST = 4.0  # Distance to detect overcrowding
SEPARATION_FORCE = 0.5 # Strength of "personal space" (prevents stampede)
TARGET_FORCE = 0.1     # Strength of pull towards safe zone
MAX_SPEED = 1.5

# Approximate Coordinates for Harrods 4th Floor (0-100 scale)
# Roughly matches the angled shape in your image
FLOOR_BOUNDARY = [
    (15, 15),  # Bottom Left
    (85, 15),  # Bottom Right
    (85, 90),  # Top Right
    (60, 95),  # Top Peak
    (15, 75),  # Top Left (Angled)
    (15, 15)   # Close loop
]

# The "Wellness Clinic" - Designated Low Density/Safe Zone
# Based on the purple area in your map
SAFE_ZONE_RECT = {'x': 25, 'y': 55, 'w': 25, 'h': 25} 

# Entrances (Spawning points)
ENTRANCES = [
    (50, 20), # Bottom Center (Women's Contemporary)
    (80, 50)  # Right Side (Childrenswear/Escalators)
]

class CrowdSimulation:
    def __init__(self):
        self.num_agents = NUM_PEOPLE
        # Random positions to start
        self.pos = np.random.rand(self.num_agents, 2) * 40 + 30 
        self.vel = (np.random.rand(self.num_agents, 2) - 0.5) * 2
        self.invacuating = False # State trigger
        
        # Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(8, 10))
        self.setup_environment()
        
        # Agents scatter plot
        self.scat = self.ax.scatter(self.pos[:, 0], self.pos[:, 1], c='blue', s=30, alpha=0.7)
        self.title = self.ax.set_title("Status: Normal Shopping")

    def setup_environment(self):
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 110)
        self.ax.set_aspect('equal')
        
        # Draw Floor Boundary
        codes = [Path.MOVETO] + [Path.LINETO] * (len(FLOOR_BOUNDARY) - 2) + [Path.CLOSEPOLY]
        path = Path(FLOOR_BOUNDARY, codes)
        patch = patches.PathPatch(path, facecolor='#fdfbf7', edgecolor='black', lw=2)
        self.ax.add_patch(patch)
        self.floor_path = path # For collision detection
        
        # Draw Wellness Clinic (Target)
        r = SAFE_ZONE_RECT
        rect = patches.Rectangle((r['x'], r['y']), r['w'], r['h'], 
                                 facecolor='#dcd0ff', edgecolor='purple', alpha=0.5, label='Wellness Clinic')
        self.ax.add_patch(rect)
        self.ax.text(r['x']+2, r['y']+r['h']-5, "Safe Zone\n(Wellness)", fontsize=9, color='purple')
        
        # Draw Departments (Visual cues only)
        self.ax.text(70, 30, "Childrenswear", fontsize=8, color='orange')
        self.ax.text(20, 25, "Women's\nContemp.", fontsize=8, color='red')
        self.ax.text(20, 50, "Burger\nBar", fontsize=8, color='green')

    def apply_forces(self):
        # 1. Separation (Anti-Stampede Logic)
        # If neighbors are too close, push away
        separation = np.zeros_like(self.pos)
        # Simple N^2 pairwise check (optimization needed for N>500)
        for i in range(self.num_agents):
            diff = self.pos - self.pos[i]
            dist = np.linalg.norm(diff, axis=1)
            # Find neighbors within range, excluding self
            mask = (dist < NEIGHBOR_DIST) & (dist > 0)
            if np.any(mask):
                # Vector pointing away from neighbors
                push = diff[mask] / dist[mask, None] 
                separation[i] = np.sum(push, axis=0)
        
        # 2. Target Attraction (Invacuation)
        target_pull = np.zeros_like(self.pos)
        if self.invacuating:
            # Center of Wellness Clinic
            tx = SAFE_ZONE_RECT['x'] + SAFE_ZONE_RECT['w']/2
            ty = SAFE_ZONE_RECT['y'] + SAFE_ZONE_RECT['h']/2
            target = np.array([tx, ty])
            
            direction = target - self.pos
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            # Normalize and apply force
            target_pull = (direction / (norm + 0.1)) * TARGET_FORCE
            
            # Stop if inside safe zone
            in_zone = (self.pos[:,0] > SAFE_ZONE_RECT['x']) & \
                      (self.pos[:,0] < SAFE_ZONE_RECT['x'] + SAFE_ZONE_RECT['w']) & \
                      (self.pos[:,1] > SAFE_ZONE_RECT['y']) & \
                      (self.pos[:,1] < SAFE_ZONE_RECT['y'] + SAFE_ZONE_RECT['h'])
            target_pull[in_zone] = 0
            self.vel[in_zone] *= 0.1 # Slow down significantly

        return (separation * SEPARATION_FORCE) + target_pull

    def update(self, frame):
        # Trigger invacuation after 50 frames
        if frame == 50:
            self.invacuating = True
            self.title.set_text("Status: INVACUATION TRIGGERED - Moving to Wellness Clinic")
            self.scat.set_color('red') # Change color to indicate alert
        
        forces = self.apply_forces()
        
        # Update Velocity and Position
        self.vel += forces
        # Cap speed
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        self.vel = np.where(speed > MAX_SPEED, self.vel / speed * MAX_SPEED, self.vel)
        
        new_pos = self.pos + self.vel * SIM_SPEED
        
        # Boundary Collision (Keep them inside the store)
        # Simple containment: if outside, push back
        in_bounds = self.floor_path.contains_points(new_pos)
        self.pos = np.where(in_bounds[:, None], new_pos, self.pos)
        # Bounce off walls (negate velocity if stuck)
        self.vel[~in_bounds] *= -0.5
        
        self.scat.set_offsets(self.pos)
        return self.scat, self.title

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=False)
        plt.show()

if __name__ == "__main__":
    sim = CrowdSimulation()
    sim.animate()
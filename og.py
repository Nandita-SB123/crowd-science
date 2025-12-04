import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.path import Path
import matplotlib.patches as patches

# --- CONFIGURATION ---
NUM_PEOPLE = 350
SIM_SPEED = 1.0
NEIGHBOR_DIST = 3.5
SEPARATION_FORCE = 0.8
TARGET_FORCE = 0.15
MAX_SPEED = 2.0
FRICTION = 0.95
SPAWN_RATE = 5  # People per frame during shopping phase

# Harrods 4th Floor Boundary (matching the floor plan shape)
FLOOR_BOUNDARY = [
    (10, 10),   # Bottom left corner
    (90, 10),   # Bottom right corner
    (90, 85),   # Top right corner
    (75, 95),   # Top right peak
    (10, 95),   # Top left
    (10, 10)    # Close loop
]

# Multiple low-density safe zones based on the floor plan
SAFE_ZONES = [
    {'x': 25, 'y': 55, 'w': 22, 'h': 25, 'name': 'Wellness Clinic', 'color': '#dcd0ff'},
    {'x': 15, 'y': 30, 'w': 20, 'h': 18, 'name': 'Burger Bar', 'color': '#d0ffdc'},
    {'x': 70, 'y': 70, 'w': 15, 'h': 18, 'name': 'Childrenswear', 'color': '#ffd0dc'}
]

# Entrances with spawn angles
ENTRANCES = [
    {'pos': (50, 12), 'angle': np.pi/2, 'name': 'Main Entrance'},
    {'pos': (88, 50), 'angle': np.pi, 'name': 'Side Entrance'},
    {'pos': (12, 70), 'angle': 0, 'name': 'West Entrance'}
]

# High-density shopping areas to avoid during invacuation
HIGH_DENSITY_ZONES = [
    {'x': 35, 'y': 15, 'w': 30, 'h': 25},  # Women's Contemporary
    {'x': 70, 'y': 35, 'w': 18, 'h': 25}   # Mini Superbrands
]

class CrowdSimulation:
    def __init__(self):
        self.max_agents = NUM_PEOPLE
        # Start with fewer people, spawn them gradually
        self.num_agents = 0
        self.pos = np.zeros((self.max_agents, 2))
        self.vel = np.zeros((self.max_agents, 2))
        self.target_zone = np.zeros(self.max_agents, dtype=int)  # Which safe zone each person targets
        self.reached_safety = np.zeros(self.max_agents, dtype=bool)
        self.spawn_counter = 0
        self.invacuating = False
        
        # Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(10, 11))
        self.setup_environment()
        
        # Agents scatter plot
        self.scat = self.ax.scatter([], [], c='blue', s=40, alpha=0.7, edgecolors='navy', linewidth=0.5)
        self.title = self.ax.set_title("Black Friday Shopping - Normal Operations", fontsize=14, fontweight='bold')
        self.info_text = self.ax.text(5, 2, "", fontsize=10)

    def setup_environment(self):
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 105)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Draw Floor Boundary
        codes = [Path.MOVETO] + [Path.LINETO] * (len(FLOOR_BOUNDARY) - 2) + [Path.CLOSEPOLY]
        path = Path(FLOOR_BOUNDARY, codes)
        patch = patches.PathPatch(path, facecolor='#f8f5f0', edgecolor='#333', lw=3)
        self.ax.add_patch(patch)
        self.floor_path = path
        
        # Draw Safe Zones
        for zone in SAFE_ZONES:
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor=zone['color'], edgecolor='darkgreen', 
                                     alpha=0.4, linewidth=2, linestyle='--')
            self.ax.add_patch(rect)
            self.ax.text(zone['x']+2, zone['y']+zone['h']-3, f"🛡️ {zone['name']}", 
                        fontsize=8, color='darkgreen', fontweight='bold')
        
        # Draw High Density Zones (subtle)
        for zone in HIGH_DENSITY_ZONES:
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor='none', edgecolor='gray', 
                                     alpha=0.3, linewidth=1, linestyle=':')
            self.ax.add_patch(rect)
        
        # Draw Entrances
        for entrance in ENTRANCES:
            circle = patches.Circle(entrance['pos'], 2, facecolor='red', edgecolor='darkred', alpha=0.6)
            self.ax.add_patch(circle)
            self.ax.text(entrance['pos'][0], entrance['pos'][1]-4, entrance['name'], 
                        fontsize=7, ha='center', color='darkred')
        
        # Labels for departments
        self.ax.text(20, 40, "Burger Bar\nArea", fontsize=9, color='green', ha='center')
        self.ax.text(50, 25, "Women's Contemporary\n& Sport", fontsize=9, color='#8b4513', ha='center')
        self.ax.text(36, 67, "Wellness\nClinic", fontsize=9, color='purple', ha='center', fontweight='bold')
        self.ax.text(77, 78, "Children's\nwear", fontsize=8, color='orange', ha='center')
        self.ax.text(78, 50, "Mini Super-\nbrands", fontsize=8, color='blue', ha='center')

    def spawn_person(self, entrance_idx):
        """Spawn a new person at an entrance"""
        if self.num_agents >= self.max_agents:
            return
        
        entrance = ENTRANCES[entrance_idx]
        # Add some randomness to spawn position
        offset = np.random.randn(2) * 1.5
        self.pos[self.num_agents] = entrance['pos'] + offset
        
        # Initial velocity in the direction of the entrance angle + randomness
        angle = entrance['angle'] + np.random.randn() * 0.3
        speed = np.random.uniform(0.5, 1.5)
        self.vel[self.num_agents] = [np.cos(angle) * speed, np.sin(angle) * speed]
        
        # Assign random target zone
        self.target_zone[self.num_agents] = np.random.randint(0, len(SAFE_ZONES))
        
        self.num_agents += 1

    def apply_forces(self):
        if self.num_agents == 0:
            return np.zeros((0, 2))
        
        forces = np.zeros((self.num_agents, 2))
        
        # 1. Separation Force (Personal Space)
        for i in range(self.num_agents):
            if self.reached_safety[i]:
                continue
                
            diff = self.pos[:self.num_agents] - self.pos[i]
            dist = np.linalg.norm(diff, axis=1)
            mask = (dist < NEIGHBOR_DIST) & (dist > 0.1)
            
            if np.any(mask):
                # Inverse square law for separation
                weights = 1.0 / (dist[mask]**2 + 0.1)
                push = diff[mask] / dist[mask, None]
                forces[i] -= np.sum(push * weights[:, None], axis=0)
        
        # 2. Target Attraction (only during invacuation)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.reached_safety[i]:
                    continue
                
                # Target the assigned safe zone
                zone = SAFE_ZONES[self.target_zone[i]]
                target = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                
                direction = target - self.pos[i]
                dist = np.linalg.norm(direction)
                
                if dist > 1.0:
                    # Add urgency that increases over time
                    urgency = 1.0 + (self.frame - 100) * 0.01
                    forces[i] += (direction / dist) * TARGET_FORCE * urgency
                
                # Check if reached safety
                if (self.pos[i, 0] > zone['x'] and 
                    self.pos[i, 0] < zone['x'] + zone['w'] and
                    self.pos[i, 1] > zone['y'] and 
                    self.pos[i, 1] < zone['y'] + zone['h']):
                    self.reached_safety[i] = True
                    self.vel[i] *= 0.05
        
        # 3. Avoid High Density Zones during invacuation
        if self.invacuating:
            for i in range(self.num_agents):
                if self.reached_safety[i]:
                    continue
                    
                for zone in HIGH_DENSITY_ZONES:
                    zone_center = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                    diff = self.pos[i] - zone_center
                    dist = np.linalg.norm(diff)
                    
                    # If close to high density zone, push away
                    if dist < 20:
                        forces[i] += (diff / (dist + 0.1)) * 0.1
        
        # 4. Random Walking (only during normal shopping)
        if not self.invacuating:
            forces += (np.random.randn(self.num_agents, 2) - 0.5) * 0.05
        
        return forces

    def update(self, frame):
        self.frame = frame
        
        # Spawn people during shopping phase
        if frame < 100 and frame % 2 == 0:
            entrance_idx = np.random.randint(0, len(ENTRANCES))
            self.spawn_person(entrance_idx)
        
        # Trigger invacuation
        if frame == 100:
            self.invacuating = True
            # self.title.set_text("⚠️ INVACUATION PROTOCOL ACTIVATED ⚠️", color='red')
            self.title.set_text("⚠️ INVACUATION PROTOCOL ACTIVATED ⚠️")
            self.title.set_fontsize(16)
        
        if self.num_agents == 0:
            return self.scat, self.title, self.info_text
        
        # Apply physics
        forces = self.apply_forces()
        self.vel[:self.num_agents] += forces
        self.vel[:self.num_agents] *= FRICTION
        
        # Cap speed
        speeds = np.linalg.norm(self.vel[:self.num_agents], axis=1, keepdims=True)
        mask = speeds.flatten() > MAX_SPEED
        self.vel[:self.num_agents][mask] = self.vel[:self.num_agents][mask] / speeds[mask] * MAX_SPEED
        
        # Update positions
        new_pos = self.pos[:self.num_agents] + self.vel[:self.num_agents] * SIM_SPEED
        
        # Boundary collision
        in_bounds = self.floor_path.contains_points(new_pos)
        self.pos[:self.num_agents] = np.where(in_bounds[:, None], new_pos, self.pos[:self.num_agents])
        self.vel[:self.num_agents][~in_bounds] *= -0.3
        
        # Update visualization
        colors = ['red' if self.invacuating and not self.reached_safety[i] else 'green' 
                 for i in range(self.num_agents)]
        self.scat.set_offsets(self.pos[:self.num_agents])
        self.scat.set_color(colors)
        
        # Update info
        safe_count = np.sum(self.reached_safety[:self.num_agents])
        self.info_text.set_text(f"People: {self.num_agents} | In Safety: {safe_count} | Frame: {frame}")
        
        return self.scat, self.title, self.info_text

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update, frames=350, 
                                      interval=50, blit=False, repeat=True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Harrods 4th Floor Invacuation Simulation")
    print("=========================================")
    print("Phase 1 (Frame 0-100): Normal Black Friday shopping")
    print("Phase 2 (Frame 100+): Invacuation protocol activated")
    print("\nMultiple safe zones: Wellness Clinic, Burger Bar, Childrenswear")
    print("People will move to nearest low-density area to prevent stampede")
    
    sim = CrowdSimulation()
    sim.animate()
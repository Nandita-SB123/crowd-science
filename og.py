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
TARGET_FORCE = 0.20  # Increased slightly to help them find zones
MAX_SPEED = 2.0
FRICTION = 0.95
SPAWN_RATE = 5 

# Harrods 4th Floor Boundary (Approximated from Image)
FLOOR_BOUNDARY = [
    (10, 10),   # Bottom left (Basil St / Hans Cres corner)
    (90, 10),   # Bottom right (Brompton Rd / Hans Cres corner)
    (90, 95),   # Top right (Brompton Rd / Hans Rd corner)
    (35, 90),   # Top Left Peak (Hans Rd slant)
    (5, 65),   # Top Left Cut (Basil St start)
    (10, 10)    # Close loop
]

# Safe Zones (Invacuation areas based on low density/enclosed spaces in map)
SAFE_ZONES = [
    # The Wellness Clinic (Top Center/Left - Purple)
    {'x': 30, 'y': 45, 'w': 25, 'h': 25, 'name': 'Wellness Clinic', 'color': '#dcd0ff'},
    
    # Gordon Ramsay Burger Bar (Left Edge - Green)
    {'x': 8, 'y': 41, 'w': 12, 'h': 17, 'name': 'Burger Bar', 'color': '#d0ffdc'},
   
    # Somewhere cafe
    {'x': 30, 'y': 70, 'w': 10, 'h': 10, 'name': 'Somewhere Cafe', 'color': "#c4ffe3"},
    
    # Georgian Restaurant
    {'x': 40, 'y': 70, 'w': 25, 'h': 20, 'name': 'Georgian Restaurant', 'color': "#c4ffe3"},
    
    # Childrenswear (Right Center - Orange/Peach)
    {'x': 65, 'y': 30, 'w': 23, 'h': 50, 'name': 'Childrenswear', 'color': '#ffe4c4'},
    
    # Toy Kingdom (Bottom Right - Cyan)
    {'x': 50, 'y': 12, 'w': 38, 'h': 15, 'name': 'Toy Kingdom', 'color': '#c4f4ff'}
    
]

# Entrances (Based on Lift/Escalator icons in map)
ENTRANCES = [
    {'pos': (45, 12), 'angle': np.pi/2, 'name': 'Hans Cres Esc.'},  # Bottom Center
    {'pos': (80, 90), 'angle': -np.pi/2 - 0.5, 'name': 'Door 10 Lifts'}, # Top Right
    {'pos': (10, 33), 'angle': 0, 'name': 'Basil St Esc.'},     # Left Side
    {'pos': (85, 20), 'angle': np.pi, 'name': 'Brompton Esc.'}  # Bottom Right
]

# High Density Shopping Areas (To avoid during invacuation)
HIGH_DENSITY_ZONES = [
    # Women's Contemporary & Sport (Bottom Left - Pink)
    {'x': 15, 'y': 15, 'w': 35, 'h': 25}, 
    
    # Mini Superbrands (Center Right - Beige)
    {'x': 65, 'y': 35, 'w': 15, 'h': 15}  
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
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_environment()
        
        # Agents scatter plot
        self.scat = self.ax.scatter([], [], c='blue', s=40, alpha=0.7, edgecolors='navy', linewidth=0.5)
        self.title = self.ax.set_title("Harrods Floor 4 - Normal Operations", fontsize=14, fontweight='bold')
        self.info_text = self.ax.text(12, 92, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    def setup_environment(self):
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Draw Floor Boundary
        codes = [Path.MOVETO] + [Path.LINETO] * (len(FLOOR_BOUNDARY) - 2) + [Path.CLOSEPOLY]
        path = Path(FLOOR_BOUNDARY, codes)
        patch = patches.PathPatch(path, facecolor='#f9f9f9', edgecolor='#333', lw=3)
        self.ax.add_patch(patch)
        self.floor_path = path
        
        # Draw Safe Zones
        for zone in SAFE_ZONES:
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor=zone['color'], edgecolor='darkgreen', 
                                     alpha=0.5, linewidth=2, linestyle='--')
            self.ax.add_patch(rect)
            # Label the safe zone
            self.ax.text(zone['x'] + zone['w']/2, zone['y'] + zone['h']/2, f"🛡️\n{zone['name']}", 
                         fontsize=8, color='darkgreen', fontweight='bold', ha='center', va='center')
        
        # Draw High Density Zones (Subtle outlines)
        for zone in HIGH_DENSITY_ZONES:
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor='none', edgecolor='#b0b0b0', 
                                     alpha=0.5, linewidth=1, linestyle=':')
            self.ax.add_patch(rect)
        
        # Draw Entrances
        for entrance in ENTRANCES:
            circle = patches.Circle(entrance['pos'], 2, facecolor='#cc3333', edgecolor='darkred', alpha=0.8)
            self.ax.add_patch(circle)
            self.ax.text(entrance['pos'][0], entrance['pos'][1]-3.5, entrance['name'], 
                         fontsize=7, ha='center', color='#cc3333', fontweight='bold')

        # Department Labels (Context)
        self.ax.text(32, 27, "Women's\nContemporary\n(High Traffic)", fontsize=8, color='#d14a6b', ha='center', alpha=0.6)

    def spawn_person(self, entrance_idx):
        if self.num_agents >= self.max_agents:
            return
        
        entrance = ENTRANCES[entrance_idx]
        offset = np.random.randn(2) * 1.5
        self.pos[self.num_agents] = entrance['pos'] + offset
        
        angle = entrance['angle'] + np.random.randn() * 0.3
        speed = np.random.uniform(0.5, 1.5)
        self.vel[self.num_agents] = [np.cos(angle) * speed, np.sin(angle) * speed]
        
        # Assign closest or random target zone? Random for now, but weighted could be better
        self.target_zone[self.num_agents] = np.random.randint(0, len(SAFE_ZONES))
        
        self.num_agents += 1

    def apply_forces(self):
        if self.num_agents == 0:
            return np.zeros((0, 2))
        
        forces = np.zeros((self.num_agents, 2))
        
        # 1. Separation Force
        for i in range(self.num_agents):
            if self.reached_safety[i]: continue
                
            diff = self.pos[:self.num_agents] - self.pos[i]
            dist = np.linalg.norm(diff, axis=1)
            mask = (dist < NEIGHBOR_DIST) & (dist > 0.1)
            
            if np.any(mask):
                # Inverse square law for separation
                weights = 1.0 / (dist[mask]**2 + 0.1)
                push = diff[mask] / dist[mask, None]
                forces[i] -= np.sum(push * weights[:, None], axis=0) * SEPARATION_FORCE
        
        # 2. Target Attraction (Invacuation)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.reached_safety[i]: continue
                
                zone = SAFE_ZONES[self.target_zone[i]]
                target = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                
                direction = target - self.pos[i]
                dist = np.linalg.norm(direction)
                
                if dist > 1.0:
                    # Add urgency that increases over time
                    urgency = 1.0 + (self.frame - 100) * 0.005
                    forces[i] += (direction / dist) * TARGET_FORCE * urgency
                
                # Check containment in zone
                if (self.pos[i, 0] > zone['x'] and 
                    self.pos[i, 0] < zone['x'] + zone['w'] and
                    self.pos[i, 1] > zone['y'] and 
                    self.pos[i, 1] < zone['y'] + zone['h']):
                    self.reached_safety[i] = True
                    self.vel[i] *= 0.05 # CHECK THIS!
        
        # 3. Avoid High Density Zones (Invacuation only)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.reached_safety[i]: continue
                for zone in HIGH_DENSITY_ZONES:
                    # # Simple bounding box avoidance
                    # if (zone['x'] - 2 < self.pos[i,0] < zone['x'] + zone['w'] + 2 and
                    #     zone['y'] - 2 < self.pos[i,1] < zone['y'] + zone['h'] + 2):
                    #     # Push away from center of density zone
                        center = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                        diff = self.pos[i] - center
                        dist = np.linalg.norm(diff)
                        forces[i] += (diff / (dist + 0.1)) * 0.1

        # 4. Random Walking (Normal Ops)
        if not self.invacuating:
            forces += (np.random.randn(self.num_agents, 2) - 0.5) * 0.05
        
        return forces

    def update(self, frame):
        self.frame = frame
        
        # Phase 1: Shopping (Spawn people)
        if frame < 200 and frame % 2 == 0:
            entrance_idx = np.random.randint(0, len(ENTRANCES))
            self.spawn_person(entrance_idx)
        
        # Phase 2: Trigger Invacuation
        if frame == 200:
            self.invacuating = True
            self.title.set_text("⚠️ INVACUATION PROTOCOL ACTIVATED ⚠️")
            self.title.set_color('red')
        
        if self.num_agents == 0:
            return self.scat, self.title, self.info_text
        
        # Physics Step
        forces = self.apply_forces()
        self.vel[:self.num_agents] += forces
        self.vel[:self.num_agents] *= FRICTION
        
        # Speed Cap
        speeds = np.linalg.norm(self.vel[:self.num_agents], axis=1, keepdims=True)
        mask = speeds.flatten() > MAX_SPEED
        self.vel[:self.num_agents][mask] = self.vel[:self.num_agents][mask] / speeds[mask] * MAX_SPEED
        
        # Position Update
        new_pos = self.pos[:self.num_agents] + self.vel[:self.num_agents] * SIM_SPEED
        
        # Wall Collision
        in_bounds = self.floor_path.contains_points(new_pos)
        self.pos[:self.num_agents] = np.where(in_bounds[:, None], new_pos, self.pos[:self.num_agents])
        self.vel[:self.num_agents][~in_bounds] *= -0.5 # Bounce off walls
        
        # Visualization Update
        colors = []
        for i in range(self.num_agents):
            if self.reached_safety[i]:
                colors.append('#32CD32') # Lime Green for safe
            elif self.invacuating:
                colors.append('#FF4500') # Orange Red for panic
            else:
                colors.append('#1E90FF') # Dodger Blue for shoppers
                
        self.scat.set_offsets(self.pos[:self.num_agents])
        self.scat.set_color(colors)
        
        safe_count = np.sum(self.reached_safety[:self.num_agents])
        self.info_text.set_text(f"Shoppers: {self.num_agents}\nIn Safe Zones: {safe_count}\nStatus: {'INVACUATING' if self.invacuating else 'SHOPPING'}")
        
        return self.scat, self.title, self.info_text

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update, frames=400, 
                                     interval=30, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = CrowdSimulation()
    sim.animate()
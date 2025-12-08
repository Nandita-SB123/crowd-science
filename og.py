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
TARGET_FORCE = 0.25
# MAX_SPEED removed here, defined per-agent type below
FRICTION = 0.95
SPAWN_RATE = 5

# --- AGENT TYPES CONFIG ---
CHILD_RATIO = 0.25   # 25% of agents will be children
ADULT_SPEED = 2.0    # Max speed for adults
CHILD_SPEED = 1.4    # Max speed for children (slower)
ADULT_SIZE = 50      # Scatter plot marker size
CHILD_SIZE = 20      # Smaller marker size

# Harrods 4th Floor Boundary (Approximated from Image)
FLOOR_BOUNDARY = [
    (10, 10),   # Bottom left (Basil St / Hans Cres corner)
    (90, 10),   # Bottom right (Brompton Rd / Hans Cres corner)
    (90, 95),   # Top right (Brompton Rd / Hans Rd corner)
    (35, 90),   # Top Left Peak (Hans Rd slant)
    (5, 65),    # Top Left Cut (Basil St start)
    (10, 10)    # Close loop
]

# Safe Zones (Invacuation areas based on low density/enclosed spaces in map)
SAFE_ZONES = [
    # The Wellness Clinic (Top Center/Left - Purple)
    {'x': 30, 'y': 45, 'w': 25, 'h': 25, 'name': 'Wellness Clinic', 'color': '#dcd0ff', 'capacity': 60},
    
    # Gordon Ramsay Burger Bar (Left Edge - Green)
    {'x': 8, 'y': 41, 'w': 12, 'h': 17, 'name': 'Burger Bar', 'color': '#d0ffdc', 'capacity': 35},
   
    # Somewhere cafe
    {'x': 30, 'y': 70, 'w': 10, 'h': 10, 'name': 'Somewhere Cafe', 'color': "#c4ffe3", 'capacity': 20},
    
    # Georgian Restaurant
    {'x': 40, 'y': 70, 'w': 25, 'h': 20, 'name': 'Georgian Restaurant', 'color': "#c4ffe3", 'capacity': 80},
    
    # Childrenswear (Right Center - Orange/Peach)
    {'x': 65, 'y': 30, 'w': 23, 'h': 50, 'name': 'Childrenswear', 'color': '#ffe4c4', 'capacity': 100},
    
    # Toy Kingdom (Bottom Right - Cyan)
    {'x': 50, 'y': 12, 'w': 38, 'h': 15, 'name': 'Toy Kingdom', 'color': '#c4f4ff', 'capacity': 80}
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
    {'x': 15, 'y': 25, 'w': 35, 'h': 25}, 
    
    # Mini Superbrands (Center Right - Beige)
    {'x': 65, 'y': 35, 'w': 15, 'h': 15}  
]

class CrowdSimulation:
    def __init__(self):
        self.max_agents = NUM_PEOPLE
        self.num_agents = 0
        self.pos = np.zeros((self.max_agents, 2))
        self.vel = np.zeros((self.max_agents, 2))
        
        # New arrays for agent types and varying speeds
        self.agent_types = np.zeros(self.max_agents, dtype=int) # 0=Adult, 1=Child
        self.max_speeds = np.zeros(self.max_agents)
        
        self.target_zone = np.zeros(self.max_agents, dtype=int)
        self.reached_safety = np.zeros(self.max_agents, dtype=bool)
        self.zone_counts = np.zeros(len(SAFE_ZONES), dtype=int)  # Track zone occupancy
        self.spawn_counter = 0
        self.invacuating = False
        
        # Setup Figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_environment()
        
        # Agents scatter plot - Initialize with empty sizes list
        self.scat = self.ax.scatter([], [], c='blue', s=[], alpha=0.7, edgecolors='navy', linewidth=0.5)
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

    def get_nearest_valid_position(self, pos):
        """Move position inside boundary if it's outside"""
        center = np.array([50, 50])
        direction = center - pos
        test_pos = pos.copy()
        step_size = 0.5
        
        for _ in range(20):  # Max iterations
            if self.floor_path.contains_point(test_pos):
                return test_pos
            test_pos += direction / np.linalg.norm(direction) * step_size
        
        return center  # Fallback to center

    def assign_zone_balanced(self, person_idx):
        """Assign person to least crowded zone based on capacity"""
        # Calculate utilization ratio for each zone
        utilization = self.zone_counts / np.array([z['capacity'] for z in SAFE_ZONES])
        
        # Add distance factor - prefer closer zones but prioritize balance
        person_pos = self.pos[person_idx]
        distances = []
        for zone in SAFE_ZONES:
            zone_center = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
            dist = np.linalg.norm(person_pos - zone_center)
            distances.append(dist)
        
        distances = np.array(distances)
        norm_distances = distances / (np.max(distances) + 0.01)  # Normalize + avoid div zero
        
        # Combined score: 70% utilization, 30% distance
        scores = 0.7 * utilization + 0.3 * norm_distances
        
        # Choose zone with lowest score
        chosen_zone = np.argmin(scores)
        self.zone_counts[chosen_zone] += 1
        return chosen_zone

    def spawn_person(self, entrance_idx):
        if self.num_agents >= self.max_agents:
            return
        
        entrance = ENTRANCES[entrance_idx]
        offset = np.random.randn(2) * 1.0  # Reduced spread
        new_pos = entrance['pos'] + offset
        
        # Ensure spawn position is valid
        if not self.floor_path.contains_point(new_pos):
            new_pos = self.get_nearest_valid_position(new_pos)
        
        self.pos[self.num_agents] = new_pos
        
        # --- DETERMINE AGENT TYPE ---
        angle = entrance['angle'] + np.random.randn() * 0.3
        
        if np.random.random() < CHILD_RATIO:
            # It's a child
            self.agent_types[self.num_agents] = 1
            self.max_speeds[self.num_agents] = CHILD_SPEED
            init_speed = np.random.uniform(0.4, 1.0) # Slower start
        else:
            # It's an adult
            self.agent_types[self.num_agents] = 0
            self.max_speeds[self.num_agents] = ADULT_SPEED
            init_speed = np.random.uniform(0.7, 1.5) # Faster start

        self.vel[self.num_agents] = [np.cos(angle) * init_speed, np.sin(angle) * init_speed]
        
        # Will assign zone during invacuation
        self.target_zone[self.num_agents] = 0
        
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
                    urgency = 1.0 + (self.frame - 200) * 0.005
                    forces[i] += (direction / dist) * TARGET_FORCE * urgency
                
                # Check containment in zone
                if (self.pos[i, 0] > zone['x'] and 
                    self.pos[i, 0] < zone['x'] + zone['w'] and
                    self.pos[i, 1] > zone['y'] and 
                    self.pos[i, 1] < zone['y'] + zone['h']):
                    self.reached_safety[i] = True
                    self.vel[i] *= 0.05
        
        # 3. Avoid High Density Zones (Invacuation only)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.reached_safety[i]: continue
                for zone in HIGH_DENSITY_ZONES:
                    center = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                    diff = self.pos[i] - center
                    dist = np.linalg.norm(diff)
                    if dist < 20:  # Within influence range
                        forces[i] += (diff / (dist + 0.1)) * 0.15

        # 4. Random Walking (Normal Ops)
        if not self.invacuating:
            forces += (np.random.randn(self.num_agents, 2) - 0.5) * 0.05
        
        return forces

    def update(self, frame):
        self.frame = frame
        
        # Phase 1: Shopping (Spawn people)
        if frame < 350 and frame % 2 == 0:
            entrance_idx = np.random.randint(0, len(ENTRANCES))
            self.spawn_person(entrance_idx)
        
        # Phase 2: Trigger Invacuation
        if frame == 350:
            self.invacuating = True
            self.title.set_text("⚠️ INVACUATION PROTOCOL ACTIVATED ⚠️")
            self.title.set_color('red')
            # Assign zones to all people using balanced algorithm
            for i in range(self.num_agents):
                self.target_zone[i] = self.assign_zone_balanced(i)
        
        if self.num_agents == 0:
            return self.scat, self.title, self.info_text
        
        # Physics Step
        forces = self.apply_forces()
        self.vel[:self.num_agents] += forces
        self.vel[:self.num_agents] *= FRICTION
        
        # --- INDIVIDUAL SPEED CAPS ---
        # Loop approach for clarity and safety with differing limits
        for i in range(self.num_agents):
            speed = np.linalg.norm(self.vel[i])
            if speed > self.max_speeds[i] and speed > 0:
                 # Normalize velocity vector and multiply by agent's specific max speed
                 self.vel[i] = (self.vel[i] / speed) * self.max_speeds[i]

        
        # Position Update
        new_pos = self.pos[:self.num_agents] + self.vel[:self.num_agents] * SIM_SPEED
        
        # Wall Collision - Better handling
        in_bounds = self.floor_path.contains_points(new_pos)
        
        for i in range(self.num_agents):
            if in_bounds[i]:
                self.pos[i] = new_pos[i]
            else:
                # Stay at old position and bounce
                self.vel[i] *= -0.5
                # Also try to nudge toward center
                center = np.array([50, 50])
                toward_center = (center - self.pos[i]) / np.linalg.norm(center - self.pos[i] + 0.01)
                self.vel[i] += toward_center * 0.2
        
        # Visualization Update
        colors = []
        sizes = [] # List to hold sizes for this frame

        for i in range(self.num_agents):
            # --- DETERMINE SIZE based on type ---
            if self.agent_types[i] == 1: # Child
                sizes.append(CHILD_SIZE)
            else: # Adult
                sizes.append(ADULT_SIZE)

            # Determine color based on state
            if self.reached_safety[i]:
                colors.append('#32CD32')
            elif self.invacuating:
                colors.append('#FF4500')
            else:
                colors.append('#1E90FF')
                
        self.scat.set_offsets(self.pos[:self.num_agents])
        self.scat.set_color(colors)
        self.scat.set_sizes(sizes) # Apply the sizes
        
        safe_count = np.sum(self.reached_safety[:self.num_agents])
        
        # Show zone distribution
        zone_info = "\n".join([f"{z['name']}: {self.zone_counts[i]}/{z['capacity']}" 
                               for i, z in enumerate(SAFE_ZONES)])
        
        self.info_text.set_text(f"Agents: {self.num_agents}\n(Adults/Children mix)\nIn Safe Zones: {safe_count}\n"
                                # f"Status: {'INVACUATING' if self.invacuating else 'SHOPPING'}\n\n{zone_info}"
                               )
        
        return self.scat, self.title, self.info_text

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update, frames=400, 
                                       interval=30, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = CrowdSimulation()
    sim.animate()
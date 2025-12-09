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
FRICTION = 0.95
SPAWN_RATE = 5

# --- CROWD BEHAVIOR CONFIG ---
COMPLIANCE_RATE = 0.65  # Only 65% of people will go to safe zones (the rest stay)
CHILD_RATIO = 0.25     # 25% of agents will be children
ADULT_SPEED = 2.0      # Max speed for adults
CHILD_SPEED = 1.4      # Max speed for children
ADULT_SIZE = 50        
CHILD_SIZE = 20        

# Harrods 4th Floor Boundary
FLOOR_BOUNDARY = [
    (10, 10), (90, 10), (90, 95), (35, 90), (5, 65), (10, 10)
]

# Safe Zones
SAFE_ZONES = [
    {'x': 30, 'y': 45, 'w': 25, 'h': 25, 'name': 'Wellness Clinic', 'color': '#dcd0ff', 'capacity': 60},
    {'x': 8, 'y': 46, 'w': 12, 'h': 17, 'name': 'Burger Bar', 'color': '#d0ffdc', 'capacity': 35},
    {'x': 30, 'y': 70, 'w': 10, 'h': 10, 'name': 'Somewhere Cafe', 'color': "#c4ffe3", 'capacity': 20},
    {'x': 40, 'y': 70, 'w': 25, 'h': 20, 'name': 'Georgian Rest.', 'color': "#c4ffe3", 'capacity': 80},
    # {'x': 65, 'y': 30, 'w': 23, 'h': 50, 'name': 'Childrenswear', 'color': '#ffe4c4', 'capacity': 100},
    {'x': 50, 'y': 12, 'w': 38, 'h': 15, 'name': 'Toys', 'color': '#c4f4ff', 'capacity': 80}
]

# Entrances
ENTRANCES = [
    {'pos': (45, 12), 'angle': np.pi/2, 'name': 'Hans Cres Esc.'},
    {'pos': (80, 90), 'angle': -np.pi/2 - 0.5, 'name': 'Door 10 Lifts'},
    {'pos': (10, 33), 'angle': 0, 'name': 'Basil St Esc.'},
    {'pos': (85, 20), 'angle': np.pi, 'name': 'Brompton Esc.'}
]

# High Density Shopping Areas (To avoid during invacuation)
HIGH_DENSITY_ZONES = [
    # Women's Contemporary & Sport (Bottom Left - Pink)
    {'x': 15, 'y': 15, 'w': 35, 'h': 29}, 
    
    # Mini Superbrands (Center Right - Beige)
    {'x': 50, 'y': 27, 'w': 15, 'h': 17} ,
    
    # Childrenswear (Right Side - Light Blue)
    {'x': 65, 'y': 30, 'w': 23, 'h': 50}
]

class CrowdSimulation:
    def __init__(self):
        self.max_agents = NUM_PEOPLE
        self.num_agents = 0
        self.pos = np.zeros((self.max_agents, 2))
        self.vel = np.zeros((self.max_agents, 2))
        
        self.agent_types = np.zeros(self.max_agents, dtype=int) # 0=Adult, 1=Child
        self.max_speeds = np.zeros(self.max_agents)
        
        # -1 implies NO target (staying put/shopping)
        self.target_zone = np.full(self.max_agents, -1, dtype=int)
        
        self.reached_safety = np.zeros(self.max_agents, dtype=bool)
        self.zone_counts = np.zeros(len(SAFE_ZONES), dtype=int)
        self.initial_zone_visitors = np.zeros(self.max_agents, dtype=bool) # Track people who started in safe zones 
        self.spawn_counter = 0
        self.invacuating = False
        
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.setup_environment()
        
        self.scat = self.ax.scatter([], [], c='blue', s=[], alpha=0.7, edgecolors='navy', linewidth=0.5)
        self.title = self.ax.set_title("Harrods Floor 4 - Normal Operations", fontsize=14, fontweight='bold')
        self.info_text = self.ax.text(12, 92, "", fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

    def setup_environment(self):
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        codes = [Path.MOVETO] + [Path.LINETO] * (len(FLOOR_BOUNDARY) - 2) + [Path.CLOSEPOLY]
        path = Path(FLOOR_BOUNDARY, codes)
        patch = patches.PathPatch(path, facecolor='#f9f9f9', edgecolor='#333', lw=3)
        self.ax.add_patch(patch)
        self.floor_path = path
        
        for zone in SAFE_ZONES:
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor=zone['color'], edgecolor='darkgreen', 
                                     alpha=0.5, linewidth=2, linestyle='--')
            self.ax.add_patch(rect)
            self.ax.text(zone['x'] + zone['w']/2, zone['y'] + zone['h']/2, f"🛡️\n{zone['name']}", 
                         fontsize=8, color='darkgreen', fontweight='bold', ha='center', va='center')
        
        for i, zone in enumerate(HIGH_DENSITY_ZONES):
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor='#ffebcc', edgecolor='#ff8800', 
                                     alpha=0.3, linewidth=2, linestyle=':')
            self.ax.add_patch(rect)
            
            # Add labels
            if i == 0:
                self.ax.text(zone['x'] + zone['w']/2, zone['y'] + zone['h']/2, 
                           "Women's\nContemporary\n& Sport", 
                           fontsize=9, color='#cc6600', ha='center', va='center', fontweight='bold')
            elif i == 1:
                self.ax.text(zone['x'] + zone['w']/2, zone['y'] + zone['h']/2, 
                           "Mini\nSuperbrands", 
                           fontsize=9, color='#cc6600', ha='center', va='center', fontweight='bold')
            elif i == 2:
                self.ax.text(zone['x'] + zone['w']/2, zone['y'] + zone['h']/2, 
                           "Childrenswear", 
                           fontsize=9, color='#cc6600', ha='center', va='center', fontweight='bold')
        
        for entrance in ENTRANCES:
            circle = patches.Circle(entrance['pos'], 2, facecolor='#cc3333', edgecolor='darkred', alpha=0.8)
            self.ax.add_patch(circle)
            self.ax.text(entrance['pos'][0], entrance['pos'][1]-3.5, entrance['name'], 
                         fontsize=7, ha='center', color='#cc3333', fontweight='bold')

    def get_nearest_valid_position(self, pos):
        center = np.array([50, 50])
        direction = center - pos
        test_pos = pos.copy()
        step_size = 0.5
        for _ in range(20):
            if self.floor_path.contains_point(test_pos):
                return test_pos
            test_pos += direction / np.linalg.norm(direction) * step_size
        return center

    def assign_zone_balanced(self, person_idx):
        utilization = self.zone_counts / np.array([z['capacity'] for z in SAFE_ZONES])
        person_pos = self.pos[person_idx]
        distances = []
        for zone in SAFE_ZONES:
            zone_center = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
            dist = np.linalg.norm(person_pos - zone_center)
            distances.append(dist)
        
        distances = np.array(distances)
        norm_distances = distances / (np.max(distances) + 0.01)
        scores = 0.7 * utilization + 0.3 * norm_distances
        chosen_zone = np.argmin(scores)
        self.zone_counts[chosen_zone] += 1
        return chosen_zone

    def spawn_person(self, entrance_idx, destination_type='shopping'):
        if self.num_agents >= self.max_agents: return
        
        entrance = ENTRANCES[entrance_idx]
        offset = np.random.randn(2) * 1.0
        new_pos = entrance['pos'] + offset
        
        if not self.floor_path.contains_point(new_pos):
            new_pos = self.get_nearest_valid_position(new_pos)
        
        # Determine if child based on destination
        if destination_type == 'childrenswear':
            # 70% chance of having a child in childrenswear area
            is_child = np.random.random() < 0.7
        elif destination_type == 'safe_zone':
            # 20% chance of child for safe zones (restaurants/cafes)
            is_child = np.random.random() < 0.2
        else:
            # Normal distribution for shopping areas
            is_child = np.random.random() < CHILD_RATIO
        
        self.pos[self.num_agents] = new_pos
        angle = entrance['angle'] + np.random.randn() * 0.3
        
        if is_child:
            self.agent_types[self.num_agents] = 1
            self.max_speeds[self.num_agents] = CHILD_SPEED
            init_speed = np.random.uniform(0.4, 1.0)
        else:
            self.agent_types[self.num_agents] = 0
            self.max_speeds[self.num_agents] = ADULT_SPEED
            init_speed = np.random.uniform(0.7, 1.5)

        self.vel[self.num_agents] = [np.cos(angle) * init_speed, np.sin(angle) * init_speed]
        
        # Handle safe zone visitors
        if destination_type == 'safe_zone':
            # Pick a random safe zone to visit
            zone_idx = np.random.randint(0, len(SAFE_ZONES))
            zone = SAFE_ZONES[zone_idx]
            # Spawn directly in or near the safe zone
            zone_pos = np.array([
                zone['x'] + np.random.uniform(0.2, 0.8) * zone['w'],
                zone['y'] + np.random.uniform(0.2, 0.8) * zone['h']
            ])
            self.pos[self.num_agents] = zone_pos
            self.vel[self.num_agents] *= 0.2  # Move slowly in safe zones
            self.initial_zone_visitors[self.num_agents] = True
            self.reached_safety[self.num_agents] = True  # They're already in a safe zone
        else:
            self.target_zone[self.num_agents] = -1 
        
        self.num_agents += 1
    
    def apply_forces(self):
        if self.num_agents == 0: return np.zeros((0, 2))
        forces = np.zeros((self.num_agents, 2))
        
        # 1. Separation
        for i in range(self.num_agents):
            if self.reached_safety[i]: continue
            diff = self.pos[:self.num_agents] - self.pos[i]
            dist = np.linalg.norm(diff, axis=1)
            mask = (dist < NEIGHBOR_DIST) & (dist > 0.1)
            if np.any(mask):
                weights = 1.0 / (dist[mask]**2 + 0.1)
                push = diff[mask] / dist[mask, None]
                forces[i] -= np.sum(push * weights[:, None], axis=0) * SEPARATION_FORCE
        
        # 2. Target Attraction (ONLY IF assigned a zone)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.reached_safety[i] or self.target_zone[i] == -1: 
                    continue
                
                zone = SAFE_ZONES[self.target_zone[i]]
                target = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                direction = target - self.pos[i]
                dist = np.linalg.norm(direction)
                
                if dist > 1.0:
                    urgency = 1.0 + (self.frame - 350) * 0.005
                    forces[i] += (direction / dist) * TARGET_FORCE * urgency
                
                if (self.pos[i, 0] > zone['x'] and self.pos[i, 0] < zone['x'] + zone['w'] and
                    self.pos[i, 1] > zone['y'] and self.pos[i, 1] < zone['y'] + zone['h']):
                    self.reached_safety[i] = True
                    self.vel[i] *= 0.05
        
        # 3. High Density Avoidance (Only for evacuating people)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.reached_safety[i] or self.target_zone[i] == -1: 
                    continue
                for zone in HIGH_DENSITY_ZONES:
                    center = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                    diff = self.pos[i] - center
                    dist = np.linalg.norm(diff)
                    if dist < 20:
                        forces[i] += (diff / (dist + 0.1)) * 0.15

        # 4. Shopping behavior - attraction to high density zones
        for i in range(self.num_agents):
            # Skip people who started in safe zones - they stay put
            if self.initial_zone_visitors[i]:
                continue
                
            # Apply to people who are staying (not evacuating)
            if not self.invacuating or self.target_zone[i] == -1:
                if not self.reached_safety[i]:
                    # Determine preferred shopping area based on whether they have children
                    if self.agent_types[i] == 1:  # If this is a child
                        # Children prefer childrenswear area
                        target_zone = HIGH_DENSITY_ZONES[2]  # Childrenswear
                    else:
                        # Adults without explicitly being with children prefer other areas
                        # 60% women's wear, 30% mini superbrands, 10% childrenswear
                        rand = np.random.random()
                        if rand < 0.6:
                            target_zone = HIGH_DENSITY_ZONES[0]  # Women's Contemporary
                        elif rand < 0.9:
                            target_zone = HIGH_DENSITY_ZONES[1]  # Mini Superbrands
                        else:
                            target_zone = HIGH_DENSITY_ZONES[2]  # Childrenswear
                    
                    center = np.array([target_zone['x'] + target_zone['w']/2, 
                                    target_zone['y'] + target_zone['h']/2])
                    direction = center - self.pos[i]
                    dist = np.linalg.norm(direction)
                    
                    if dist > 5:  # Only attract if they're far away
                        forces[i] += (direction / dist) * 0.15
                    
                    # Add random wandering
                    forces[i] += (np.random.randn(2)) * 0.3
            
        return forces

    def update(self, frame):
        self.frame = frame
        
        # Phase 1: Shopping
        if frame < 350:
            if frame % 2 == 0:
                # 80% regular shoppers, 15% go to safe zones, 5% go to childrenswear
                rand = np.random.random()
                if rand < 0.15:
                    entrance_idx = np.random.randint(0, len(ENTRANCES))
                    self.spawn_person(entrance_idx, 'safe_zone')
                elif rand < 0.20:
                    entrance_idx = np.random.randint(0, len(ENTRANCES))
                    self.spawn_person(entrance_idx, 'childrenswear')
                else:
                    entrance_idx = np.random.randint(0, len(ENTRANCES))
                    self.spawn_person(entrance_idx, 'shopping')
        
        # Phase 2: Trigger Invacuation
        if frame == 350:
            self.invacuating = True
            self.title.set_text("⚠️ PARTIAL INVACUATION (DENSITY REDUCTION) ⚠️")
            self.title.set_color('red')
            
            # Select only a % of people to move
            for i in range(self.num_agents):
                # If random number is < compliance rate, they move. Else, they stay (-1)
                if np.random.random() < COMPLIANCE_RATE:
                    self.target_zone[i] = self.assign_zone_balanced(i)
                else:
                    self.target_zone[i] = -1 # Explicitly set to stay
        
        if self.num_agents == 0: return self.scat, self.title, self.info_text
        
        # Physics
        forces = self.apply_forces()
        self.vel[:self.num_agents] += forces
        self.vel[:self.num_agents] *= FRICTION
        
        for i in range(self.num_agents):
            speed = np.linalg.norm(self.vel[i])
            # People in safe zones move very slowly
            max_speed = self.max_speeds[i]
            if self.initial_zone_visitors[i]:
                max_speed *= 0.3
            
            if speed > max_speed and speed > 0:
                self.vel[i] = (self.vel[i] / speed) * max_speed

        new_pos = self.pos[:self.num_agents] + self.vel[:self.num_agents] * SIM_SPEED
        in_bounds = self.floor_path.contains_points(new_pos)
        
        for i in range(self.num_agents):
            if in_bounds[i]:
                self.pos[i] = new_pos[i]
            else:
                self.vel[i] *= -0.5
                center = np.array([50, 50])
                toward_center = (center - self.pos[i]) / np.linalg.norm(center - self.pos[i] + 0.01)
                self.vel[i] += toward_center * 0.2
        
        # Visuals
        colors = []
        sizes = []
        for i in range(self.num_agents):
            if self.agent_types[i] == 1: sizes.append(CHILD_SIZE)
            else: sizes.append(ADULT_SIZE)

            if self.reached_safety[i]:
                colors.append('#32CD32') # Safe (Green)
            elif self.invacuating and self.target_zone[i] != -1:
                colors.append('#FF4500') # Invacuating (Orange/Red)
            else:
                colors.append('#1E90FF') # Staying/Shopping (Blue)
                
        self.scat.set_offsets(self.pos[:self.num_agents])
        self.scat.set_color(colors)
        self.scat.set_sizes(sizes)
        
        safe_count = np.sum(self.reached_safety[:self.num_agents])
        staying_count = np.sum(self.target_zone[:self.num_agents] == -1)
        
        self.info_text.set_text(f"Total: {self.num_agents}\nInvacuating: {self.num_agents - staying_count}\nStaying: {staying_count}\nSafe: {safe_count}")
        
        return self.scat, self.title, self.info_text

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update, frames=600, 
                                       interval=30, blit=False, repeat=False)
        # anim.save('crowd_simulation.gif', writer='ffmpeg', fps=30)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = CrowdSimulation()
    sim.animate()
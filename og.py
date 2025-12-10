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
COMPLIANCE_RATE = 0.65  # Only 65% of people will relocate to low-density zones
CHILD_RATIO = 0.25     # 25% of agents will be children
ADULT_SPEED = 1.34      # Max speed for adults
CHILD_SPEED = 1.05      # Max speed for children
ADULT_SIZE = 50        
CHILD_SIZE = 20

# --- DENSITY-BASED SLOWDOWN CONFIG ---
DENSITY_RADIUS = 5.0    # Check density within this radius
LOW_DENSITY_THRESHOLD = 3   # Less than 3 people nearby = free movement
HIGH_DENSITY_THRESHOLD = 10  # More than 10 people nearby = significant slowdown
MIN_SPEED_MULTIPLIER = 0.3   # Slowest speed is 30% of max speed

# --- REAL-WORLD SCALING ---
FLOOR_AREA_SQFT = 50000  # Real floor area
SIMULATION_SCALE = 100   # Our simulation is 100x100 units
SQFT_PER_UNIT = FLOOR_AREA_SQFT / (SIMULATION_SCALE ** 2)  # sq ft per simulation unit²

# Harrods 4th Floor Boundary
FLOOR_BOUNDARY = [
    (10, 10), (90, 10), (90, 95), (35, 90), (5, 65), (10, 10)
]

# Low Density Zones (destinations for density reduction)
LOW_DENSITY_ZONES = [
    {'x': 30, 'y': 45, 'w': 25, 'h': 25, 'name': 'Wellness Clinic', 'color': '#dcd0ff', 'capacity': 60},
    {'x': 8, 'y': 46, 'w': 12, 'h': 17, 'name': 'Burger Bar', 'color': '#d0ffdc', 'capacity': 35},
    {'x': 30, 'y': 70, 'w': 10, 'h': 10, 'name': 'Somewhere Cafe', 'color': "#c4ffe3", 'capacity': 20},
    {'x': 40, 'y': 70, 'w': 25, 'h': 20, 'name': 'Georgian Rest.', 'color': "#c4ffe3", 'capacity': 80},
    {'x': 50, 'y': 12, 'w': 38, 'h': 15, 'name': 'Toys', 'color': '#c4f4ff', 'capacity': 80}
]

# Entrances
ENTRANCES = [
    {'pos': (45, 12), 'angle': np.pi/2, 'name': 'Hans Cres Esc.'},
    {'pos': (80, 90), 'angle': -np.pi/2 - 0.5, 'name': 'Door 10 Lifts'},
    {'pos': (10, 33), 'angle': 0, 'name': 'Basil St Esc.'},
    {'pos': (85, 20), 'angle': np.pi, 'name': 'Brompton Esc.'}
]

# High Density Shopping Areas
HIGH_DENSITY_ZONES = [
    {'x': 15, 'y': 15, 'w': 35, 'h': 29}, 
    {'x': 50, 'y': 27, 'w': 15, 'h': 17},
    {'x': 65, 'y': 30, 'w': 23, 'h': 50}
]

class CrowdSimulation:
    def __init__(self):
        self.max_agents = NUM_PEOPLE
        self.num_agents = 0
        self.pos = np.zeros((self.max_agents, 2))
        self.vel = np.zeros((self.max_agents, 2))
        
        self.agent_types = np.zeros(self.max_agents, dtype=int)
        self.max_speeds = np.zeros(self.max_agents)
        self.target_zone = np.full(self.max_agents, -1, dtype=int)
        self.parent_child_groups = {}  # Maps child index to parent index
        
        self.in_low_density_zone = np.zeros(self.max_agents, dtype=bool)
        self.zone_counts = np.zeros(len(LOW_DENSITY_ZONES), dtype=int)
        self.initial_zone_visitors = np.zeros(self.max_agents, dtype=bool)
        self.spawn_counter = 0
        self.invacuating = False
        
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.setup_environment()
        
        self.scat = self.ax.scatter([], [], c='blue', s=[], alpha=0.7, edgecolors='navy', linewidth=0.5)
        self.title = self.ax.set_title("Harrods Floor 4 - Normal Operations", fontsize=14, fontweight='bold')
        self.info_text = self.ax.text(12, 92, "", fontsize=9, bbox=dict(facecolor='white', alpha=0.9))
        
        # Add simulation info text in bottom right
        self.sim_info_text = self.ax.text(88, 6, "", fontsize=8, ha='right', va='bottom',
                                          bbox=dict(facecolor='lightyellow', alpha=0.9, edgecolor='gray'))

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
        
        for zone in LOW_DENSITY_ZONES:
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor=zone['color'], edgecolor='darkgreen', 
                                     alpha=0.4, linewidth=2, linestyle='--')
            self.ax.add_patch(rect)
            self.ax.text(zone['x'] + zone['w']/2, zone['y'] + zone['h']/2, f"🏪\n{zone['name']}", 
                         fontsize=8, color='darkgreen', fontweight='bold', ha='center', va='center')
        
        for i, zone in enumerate(HIGH_DENSITY_ZONES):
            rect = patches.Rectangle((zone['x'], zone['y']), zone['w'], zone['h'], 
                                     facecolor='#ffebcc', edgecolor='#ff8800', 
                                     alpha=0.3, linewidth=2, linestyle=':')
            self.ax.add_patch(rect)
            
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

    def calculate_local_density(self, agent_idx):
        """Calculate how many people are near this agent"""
        pos = self.pos[agent_idx]
        diff = self.pos[:self.num_agents] - pos
        distances = np.linalg.norm(diff, axis=1)
        nearby = np.sum((distances < DENSITY_RADIUS) & (distances > 0.1))
        return nearby

    def get_density_speed_multiplier(self, nearby_count):
        """Returns a speed multiplier based on local density (1.0 = full speed, 0.3 = slowest)"""
        if nearby_count <= LOW_DENSITY_THRESHOLD:
            return 1.0
        elif nearby_count >= HIGH_DENSITY_THRESHOLD:
            return MIN_SPEED_MULTIPLIER
        else:
            # Linear interpolation between thresholds
            ratio = (nearby_count - LOW_DENSITY_THRESHOLD) / (HIGH_DENSITY_THRESHOLD - LOW_DENSITY_THRESHOLD)
            return 1.0 - ratio * (1.0 - MIN_SPEED_MULTIPLIER)

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
        utilization = self.zone_counts / np.array([z['capacity'] for z in LOW_DENSITY_ZONES])
        person_pos = self.pos[person_idx]
        distances = []
        for zone in LOW_DENSITY_ZONES:
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
        
        # Determine if this will be a parent-child pair
        spawn_with_child = False
        if destination_type == 'childrenswear':
            # Always spawn parent-child pairs in childrenswear
            spawn_with_child = True
        elif destination_type == 'shopping':
            # 25% chance of parent-child pair in general shopping
            spawn_with_child = np.random.random() < 0.25
        
        if spawn_with_child and self.num_agents < self.max_agents - 1:
            # Spawn parent first
            self.pos[self.num_agents] = new_pos
            angle = entrance['angle'] + np.random.randn() * 0.3
            
            self.agent_types[self.num_agents] = 0  # Adult
            self.max_speeds[self.num_agents] = ADULT_SPEED
            init_speed = np.random.uniform(0.5, 0.9)  # Parents walk slower with kids
            self.vel[self.num_agents] = [np.cos(angle) * init_speed, np.sin(angle) * init_speed]
            self.target_zone[self.num_agents] = -1
            
            parent_idx = self.num_agents
            self.num_agents += 1
            
            # Spawn child next to parent
            child_offset = np.random.randn(2) * 0.8
            child_pos = new_pos + child_offset
            if not self.floor_path.contains_point(child_pos):
                child_pos = new_pos
            
            self.pos[self.num_agents] = child_pos
            self.agent_types[self.num_agents] = 1  # Child
            self.max_speeds[self.num_agents] = CHILD_SPEED * 0.85  # Children with parents walk slower
            self.vel[self.num_agents] = self.vel[parent_idx] * 0.95  # Match parent velocity
            self.target_zone[self.num_agents] = -1
            
            # Link child to parent
            self.parent_child_groups[self.num_agents] = parent_idx
            
            self.num_agents += 1
        else:
            # Spawn single adult (or single child in low-density zones)
            if destination_type == 'low_density_zone':
                is_child = np.random.random() < 0.2
            else:
                is_child = False  # Single shoppers are adults
            
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
            
            if destination_type == 'low_density_zone':
                zone_idx = np.random.randint(0, len(LOW_DENSITY_ZONES))
                zone = LOW_DENSITY_ZONES[zone_idx]
                zone_pos = np.array([
                    zone['x'] + np.random.uniform(0.2, 0.8) * zone['w'],
                    zone['y'] + np.random.uniform(0.2, 0.8) * zone['h']
                ])
                self.pos[self.num_agents] = zone_pos
                self.vel[self.num_agents] *= 0.2
                self.initial_zone_visitors[self.num_agents] = True
                self.in_low_density_zone[self.num_agents] = True
            else:
                self.target_zone[self.num_agents] = -1 
            
            self.num_agents += 1
    
    def apply_forces(self):
        if self.num_agents == 0: return np.zeros((0, 2))
        forces = np.zeros((self.num_agents, 2))
        
        # 0. Parent-child cohesion (keep families together)
        for child_idx, parent_idx in self.parent_child_groups.items():
            if child_idx >= self.num_agents or parent_idx >= self.num_agents:
                continue
            if self.in_low_density_zone[child_idx] or self.in_low_density_zone[parent_idx]:
                continue
                
            # Pull child toward parent
            diff = self.pos[parent_idx] - self.pos[child_idx]
            dist = np.linalg.norm(diff)
            if dist > 2.0:  # If child strays too far
                forces[child_idx] += (diff / (dist + 0.01)) * 0.4
            
            # Slightly pull parent toward child (so they don't leave child behind)
            if dist > 3.0:
                forces[parent_idx] += (-diff / (dist + 0.01)) * 0.15
        
        # 1. Separation
        for i in range(self.num_agents):
            if self.in_low_density_zone[i]: continue
            diff = self.pos[:self.num_agents] - self.pos[i]
            dist = np.linalg.norm(diff, axis=1)
            mask = (dist < NEIGHBOR_DIST) & (dist > 0.1)
            if np.any(mask):
                weights = 1.0 / (dist[mask]**2 + 0.1)
                push = diff[mask] / dist[mask, None]
                forces[i] -= np.sum(push * weights[:, None], axis=0) * SEPARATION_FORCE
        
        # 2. Target Attraction (calm, no urgency)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.in_low_density_zone[i] or self.target_zone[i] == -1: 
                    continue
                
                zone = LOW_DENSITY_ZONES[self.target_zone[i]]
                target = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                direction = target - self.pos[i]
                dist = np.linalg.norm(direction)
                
                if dist > 1.0:
                    # No urgency multiplier - just calm redirection
                    forces[i] += (direction / dist) * TARGET_FORCE * 0.8
                
                if (self.pos[i, 0] > zone['x'] and self.pos[i, 0] < zone['x'] + zone['w'] and
                    self.pos[i, 1] > zone['y'] and self.pos[i, 1] < zone['y'] + zone['h']):
                    self.in_low_density_zone[i] = True
                    self.vel[i] *= 0.2  # Slow down in zone
        
        # 3. High Density Avoidance (Only for relocating people)
        if self.invacuating:
            for i in range(self.num_agents):
                if self.in_low_density_zone[i] or self.target_zone[i] == -1: 
                    continue
                for zone in HIGH_DENSITY_ZONES:
                    center = np.array([zone['x'] + zone['w']/2, zone['y'] + zone['h']/2])
                    diff = self.pos[i] - center
                    dist = np.linalg.norm(diff)
                    if dist < 20:
                        forces[i] += (diff / (dist + 0.1)) * 0.15

        # 4. Shopping behavior - attraction to high density zones
        for i in range(self.num_agents):
            if self.initial_zone_visitors[i]:
                continue
                
            if not self.invacuating or self.target_zone[i] == -1:
                if not self.in_low_density_zone[i]:
                    # Check if this person is in a parent-child group
                    is_child_with_parent = i in self.parent_child_groups
                    is_parent = i in self.parent_child_groups.values()
                    
                    if is_child_with_parent or is_parent:
                        # Families prefer childrenswear
                        target_zone = HIGH_DENSITY_ZONES[2]
                    elif self.agent_types[i] == 1:
                        # Standalone children (shouldn't happen much) go to childrenswear
                        target_zone = HIGH_DENSITY_ZONES[2]
                    else:
                        # Single adult shoppers
                        rand = np.random.random()
                        if rand < 0.6:
                            target_zone = HIGH_DENSITY_ZONES[0]
                        elif rand < 0.9:
                            target_zone = HIGH_DENSITY_ZONES[1]
                        else:
                            target_zone = HIGH_DENSITY_ZONES[2]
                    
                    center = np.array([target_zone['x'] + target_zone['w']/2, 
                                    target_zone['y'] + target_zone['h']/2])
                    direction = center - self.pos[i]
                    dist = np.linalg.norm(direction)
                    
                    if dist > 5:
                        forces[i] += (direction / dist) * 0.15
                    
                    forces[i] += (np.random.randn(2)) * 0.3
            
        return forces

    def update(self, frame):
        self.frame = frame
        
        # Phase 1: Shopping (continue spawning throughout)
        if frame < 500:
            if frame % 2 == 0:
                rand = np.random.random()
                if rand < 0.15:
                    entrance_idx = np.random.randint(0, len(ENTRANCES))
                    self.spawn_person(entrance_idx, 'low_density_zone')
                elif rand < 0.20:
                    entrance_idx = np.random.randint(0, len(ENTRANCES))
                    self.spawn_person(entrance_idx, 'childrenswear')
                else:
                    entrance_idx = np.random.randint(0, len(ENTRANCES))
                    self.spawn_person(entrance_idx, 'shopping')
        
        # Phase 2: Trigger Density Reduction
        if frame == 350:
            self.invacuating = True
            self.title.set_text("⚠️ DENSITY REDUCTION IN PROGRESS ⚠️")
            self.title.set_color('orange')
            
            # Select only a % of people to relocate
            for i in range(self.num_agents):
                if np.random.random() < COMPLIANCE_RATE and not self.initial_zone_visitors[i]:
                    self.target_zone[i] = self.assign_zone_balanced(i)
                else:
                    self.target_zone[i] = -1  # Stay in current area
        
        if self.num_agents == 0: return self.scat, self.title, self.info_text, self.sim_info_text
        
        # Physics
        forces = self.apply_forces()
        self.vel[:self.num_agents] += forces
        self.vel[:self.num_agents] *= FRICTION
        
        # Apply density-based speed limiting
        for i in range(self.num_agents):
            speed = np.linalg.norm(self.vel[i])
            max_speed = self.max_speeds[i]
            
            if self.initial_zone_visitors[i] or self.in_low_density_zone[i]:
                max_speed *= 0.3  # Move slowly in low-density zones
            else:
                # Apply density slowdown
                nearby_count = self.calculate_local_density(i)
                density_multiplier = self.get_density_speed_multiplier(nearby_count)
                max_speed *= density_multiplier
            
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

            if self.in_low_density_zone[i]:
                colors.append('#90EE90')  # Light green - in low density zone
            elif self.invacuating and self.target_zone[i] != -1:
                colors.append('#FFA500')  # Orange - relocating
            else:
                colors.append('#1E90FF')  # Blue - staying/shopping
                
        self.scat.set_offsets(self.pos[:self.num_agents])
        self.scat.set_color(colors)
        self.scat.set_sizes(sizes)
        
        relocated_count = np.sum(self.in_low_density_zone[:self.num_agents])
        staying_count = np.sum(self.target_zone[:self.num_agents] == -1)
        relocating_count = self.num_agents - staying_count - relocated_count
        
        # Calculate average density
        total_density = sum(self.calculate_local_density(i) for i in range(self.num_agents))
        avg_density = total_density / self.num_agents if self.num_agents > 0 else 0
        
        self.info_text.set_text(f"Total: {self.num_agents}\nRelocating: {relocating_count}\n"
                               f"Staying: {staying_count}\nIn Low-Density Zones: {relocated_count}\n"
                               f"Avg Density: {avg_density:.1f}")
        
        # Update simulation info
        sim_time_mins = frame / 30  # 30 frames per second
        self.sim_info_text.set_text(f"Floor Area: {FLOOR_AREA_SQFT:,} sq ft\n"
                                    f"Sim Time: {sim_time_mins:.1f} min\n"
                                    f"⏩ Time-lapse view")
        
        return self.scat, self.title, self.info_text, self.sim_info_text

    def animate(self):
        anim = animation.FuncAnimation(self.fig, self.update, frames=600, 
                                       interval=30, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = CrowdSimulation()
    sim.animate()
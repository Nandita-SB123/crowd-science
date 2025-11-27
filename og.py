import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class Agent:
    """Represents a person in the simulation"""
    id: int
    position: np.ndarray
    velocity: np.ndarray
    goal: np.ndarray
    radius: float
    mass: float
    max_speed: float
    agent_type: str  # 'adult', 'child', 'elderly', 'parent_with_stroller'
    
    def __post_init__(self):
        self.desired_speed = self.max_speed * random.uniform(0.8, 1.0)

class SocialForceModel:
    """
    Implements the Social Force Model for pedestrian dynamics
    Based on Helbing & Molnar (1995)
    """
    
    def __init__(self, room_size=(50, 50)):
        # Physical constants
        self.tau = 0.5  # Relaxation time
        self.A = 2000  # Interaction strength (repulsion from other agents)
        self.B = 0.08  # Interaction range
        self.k = 1.2e5  # Body force constant
        self.kappa = 2.4e5  # Friction force constant
        self.wall_A = 2000  # Wall repulsion strength
        self.wall_B = 0.08  # Wall repulsion range
        
        self.room_size = room_size
        self.agents: List[Agent] = []
        self.walls = []
        self.exits = []
        self.time_step = 0.05
        
    def add_agent(self, position, goal, agent_type='adult'):
        """Add an agent to the simulation"""
        # Different characteristics for different agent types
        agent_params = {
            'adult': {'radius': 0.25, 'mass': 80, 'max_speed': 10.5},
            'child': {'radius': 0.2, 'mass': 40, 'max_speed': 10.2},
            'elderly': {'radius': 0.25, 'mass': 75, 'max_speed': 9.8},
            'parent_with_stroller': {'radius': 0.4, 'mass': 100, 'max_speed': 10.0}
        }
        
        params = agent_params.get(agent_type, agent_params['adult'])
        
        agent = Agent(
            id=len(self.agents),
            position=np.array(position, dtype=float),
            velocity=np.array([0.0, 0.0]),
            goal=np.array(goal, dtype=float),
            radius=params['radius'],
            mass=params['mass'],
            max_speed=params['max_speed'],
            agent_type=agent_type
        )
        self.agents.append(agent)
        
    def add_wall(self, start, end):
        """Add a wall/obstacle"""
        self.walls.append((np.array(start), np.array(end)))
        
    def add_exit(self, position, width):
        """Add an exit point"""
        self.exits.append({'position': np.array(position), 'width': width})
    
    def driving_force(self, agent):
        """Calculate the driving force towards the goal"""
        direction = agent.goal - agent.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            return np.array([0.0, 0.0])
        
        desired_direction = direction / distance
        desired_velocity = agent.desired_speed * desired_direction
        
        return agent.mass * (desired_velocity - agent.velocity) / self.tau
    
    def agent_repulsion_force(self, agent_i, agent_j):
        """Calculate repulsion force between two agents"""
        diff = agent_i.position - agent_j.position
        distance = np.linalg.norm(diff)
        
        if distance < 0.01:
            distance = 0.01
            
        direction = diff / distance
        
        # Exponential repulsion
        force_magnitude = self.A * np.exp((agent_i.radius + agent_j.radius - distance) / self.B)
        force = force_magnitude * direction
        
        # Body force (only when overlapping)
        overlap = agent_i.radius + agent_j.radius - distance
        if overlap > 0:
            body_force = self.k * overlap * direction
            
            # Tangential friction
            tangent = np.array([-direction[1], direction[0]])
            relative_velocity = agent_j.velocity - agent_i.velocity
            tangential_velocity = np.dot(relative_velocity, tangent)
            friction_force = self.kappa * overlap * tangential_velocity * tangent
            
            force += body_force + friction_force
            
        return force
    
    def wall_repulsion_force(self, agent):
        """Calculate repulsion force from walls"""
        total_force = np.array([0.0, 0.0])
        
        for wall_start, wall_end in self.walls:
            # Calculate closest point on wall to agent
            wall_vec = wall_end - wall_start
            wall_length = np.linalg.norm(wall_vec)
            
            if wall_length < 0.01:
                continue
                
            wall_dir = wall_vec / wall_length
            
            # Project agent position onto wall
            ap = agent.position - wall_start
            t = np.dot(ap, wall_dir)
            t = np.clip(t, 0, wall_length)
            
            closest_point = wall_start + t * wall_dir
            diff = agent.position - closest_point
            distance = np.linalg.norm(diff)
            
            if distance < 0.01:
                distance = 0.01
                
            direction = diff / distance
            
            # Exponential repulsion from wall
            force_magnitude = self.wall_A * np.exp((agent.radius - distance) / self.wall_B)
            force = force_magnitude * direction
            
            # Body force if overlapping
            overlap = agent.radius - distance
            if overlap > 0:
                force += self.k * overlap * direction
                
            total_force += force
            
        return total_force
    
    def update(self):
        """Update simulation by one time step"""
        forces = []
        
        # Calculate forces for each agent
        for agent in self.agents:
            force = self.driving_force(agent)
            
            # Agent-agent repulsion
            for other in self.agents:
                if agent.id != other.id:
                    force += self.agent_repulsion_force(agent, other)
            
            # Wall repulsion
            force += self.wall_repulsion_force(agent)
            
            forces.append(force)
        
        # Update positions and velocities
        for agent, force in zip(self.agents, forces):
            acceleration = force / agent.mass
            agent.velocity += acceleration * self.time_step
            
            # Limit speed
            speed = np.linalg.norm(agent.velocity)
            if speed > agent.max_speed:
                agent.velocity = (agent.velocity / speed) * agent.max_speed
            
            agent.position += agent.velocity * self.time_step
            
            # Keep agents in bounds
            agent.position[0] = np.clip(agent.position[0], 0, self.room_size[0])
            agent.position[1] = np.clip(agent.position[1], 0, self.room_size[1])

def create_harrods_floor4_layout():
    """Create a simplified Harrods Floor 4 layout"""
    sim = SocialForceModel(room_size=(60, 40))
    
    # Add outer walls
    sim.add_wall([0, 0], [60, 0])
    sim.add_wall([60, 0], [60, 40])
    sim.add_wall([60, 40], [0, 40])
    sim.add_wall([0, 40], [0, 0])
    
    # Add internal walls/sections (simplified departments)
    # Restaurant area (top left)
    sim.add_wall([15, 30], [15, 40])
    sim.add_wall([15, 30], [25, 30])
    
    # Kids section (top right)
    sim.add_wall([40, 30], [40, 40])
    sim.add_wall([40, 30], [50, 30])
    
    # Center corridor obstacles
    sim.add_wall([25, 15], [35, 15])
    sim.add_wall([25, 25], [35, 25])
    
    # Add exits
    sim.add_exit([30, 0], 3)  # Main exit (bottom center)
    sim.add_exit([0, 20], 2)  # Side exit (left)
    sim.add_exit([60, 20], 2)  # Side exit (right)
    
    return sim

def populate_black_friday_crowd(sim, num_agents=100):
    """Add agents with Black Friday demographics"""
    
    # Demographics distribution
    agent_types = ['adult'] * 60 + ['child'] * 15 + ['elderly'] * 10 + ['parent_with_stroller'] * 15
    random.shuffle(agent_types)
    
    # Distribute agents across the floor
    for i in range(min(num_agents, len(agent_types))):
        # Random starting position
        x = random.uniform(5, 55)
        y = random.uniform(5, 35)
        
        # Assign to nearest exit as goal
        exit_pos = random.choice(sim.exits)['position']
        
        sim.add_agent(
            position=[x, y],
            goal=exit_pos,
            agent_type=agent_types[i]
        )

def visualize_simulation(sim, frames=500):
    """Create animated visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    def animate(frame):
        ax.clear()
        ax.set_xlim(0, sim.room_size[0])
        ax.set_ylim(0, sim.room_size[1])
        ax.set_aspect('equal')
        ax.set_title(f'Harrods Floor 4 Evacuation - Frame {frame}')
        
        # Draw walls
        for wall_start, wall_end in sim.walls:
            ax.plot([wall_start[0], wall_end[0]], 
                   [wall_start[1], wall_end[1]], 'k-', linewidth=2)
        
        # Draw exits
        for exit_info in sim.exits:
            exit_rect = patches.Rectangle(
                (exit_info['position'][0] - exit_info['width']/2, 
                 exit_info['position'][1] - 0.5),
                exit_info['width'], 1,
                linewidth=0, edgecolor='none', facecolor='green', alpha=0.3
            )
            ax.add_patch(exit_rect)
        
        # Draw agents
        colors = {'adult': 'blue', 'child': 'yellow', 'elderly': 'red', 
                 'parent_with_stroller': 'purple'}
        
        for agent in sim.agents:
            circle = plt.Circle(agent.position, agent.radius, 
                              color=colors.get(agent.agent_type, 'blue'), 
                              alpha=0.6)
            ax.add_patch(circle)
            
            # Draw velocity vector
            if np.linalg.norm(agent.velocity) > 0.1:
                ax.arrow(agent.position[0], agent.position[1],
                        agent.velocity[0]*0.5, agent.velocity[1]*0.5,
                        head_width=0.3, head_length=0.2, fc='black', 
                        ec='black', alpha=0.3)
        
        # Update simulation
        sim.update()
        
        # Legend
        legend_elements = [
            patches.Patch(color='blue', label='Adult'),
            patches.Patch(color='yellow', label='Child'),
            patches.Patch(color='red', label='Elderly'),
            patches.Patch(color='purple', label='Parent w/ Stroller'),
            patches.Patch(color='green', label='Exit', alpha=0.3)
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
    anim = FuncAnimation(fig, animate, frames=frames, interval=50, repeat=False)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    print("Creating Harrods Floor 4 evacuation simulation...")
    
    # Create simulation
    sim = create_harrods_floor4_layout()
    
    # Add Black Friday crowd
    populate_black_friday_crowd(sim, num_agents=80)
    
    print(f"Simulation initialized with {len(sim.agents)} agents")
    print("Starting visualization...")
    
    # Run visualization
    visualize_simulation(sim, frames=600)
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import Person

class EvacuationModel(Model):
    def __init__(self, width=30, height=30, N=200):
        self.num_agents = N
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)

        # Add agents
        for i in range(self.num_agents):
            a = Person(i, self)
            self.schedule.add(a)
            x, y = self.random.randrange(width), self.random.randrange(height)
            self.grid.place_agent(a, (x, y))

        self.running = True

    def step(self):
        self.schedule.step()

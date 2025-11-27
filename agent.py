from mesa import Agent

class Person(Agent):
    def __init__(self, unique_id, model, speed=1):
        super().__init__(unique_id, model)
        self.speed = speed

    def step(self):
        # Move towards nearest exit
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False
        )
        # Example heuristic: move to lowest distance cell
        next_moves = sorted(possible_steps, key=lambda c: self.distance_to_exit(c))
        self.model.grid.move_agent(self, next_moves[0])

    def distance_to_exit(self, pos):
        exits = [(0,0), (29,29)]  # example exits
        return min([((pos[0]-ex[0])**2 + (pos[1]-ex[1])**2)**0.5 for ex in exits])

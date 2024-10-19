import mesa
import random


class HoneyBeeAgent(mesa.Agent):
    def __init__(self, bee_agent_id, bee_model, max_nectar):
        super().__init__(bee_agent_id, bee_model)
        self.status = 's'
        self.nectar = 0
        self.max_nectar = max_nectar
        self.flower = None

    # Search for flowers in the current cell or move randomly
    def search(self):
        if self.flower is None:
            agents_at_pos = self.model.grid.get_cell_list_contents(self.pos)  # Use self.model
            available_flowers = [agent for agent in agents_at_pos if isinstance(agent, FlowerAgent)]
            if available_flowers:
                self.flower = random.choice(available_flowers)

            else:
                # Move randomly if no flower is found
                new_pos = (self.pos[0] + random.randint(-1, 1), self.pos[1] + random.randint(-1, 1))
                if not self.model.grid.out_of_bounds(new_pos):
                    self.model.grid.move_agent(self, new_pos)  # Correct agent movement

        else:
            if self.pos == self.flower.pos:
                if self.flower.nectar > 0:
                    self.nectar += 1
                    self.flower.nectar -= 1

            else:
                x_inc = 0
                y_inc = 0

                if self.pos[0] < self.flower.pos[0]:
                    x_inc = 1

                elif self.pos[0] > self.flower.pos[0]:
                    x_inc = -1

                if self.pos[1] < self.flower.pos[1]:
                    y_inc = 1

                elif self.pos[1] > self.flower.pos[1]:
                    y_inc = -1

                self.model.grid.move_agent(self, (self.pos[0] + x_inc, self.pos[1] + y_inc))



    def return_to_hive(self):
        x_move = 0
        y_move = 0

        # Navigate back to the hive
        if self.pos != self.model.hive:
            if self.pos[0] < self.model.hive[0]:
                x_move = 1
            elif self.pos[0] > self.model.hive[0]:
                x_move = -1

            if self.pos[1] < self.model.hive[1]:
                y_move = 1
            elif self.pos[1] > self.model.hive[1]:
                y_move = -1

        new_pos = (self.pos[0] + x_move, self.pos[1] + y_move)
        self.model.grid.move_agent(self, new_pos)  # Move agent to new position

    def unload(self):
        if self.nectar > 0:
            self.nectar -= 1
            self.model.hive_stock += 1  # Update the hive stock in the model

    def update_status(self):
        # Update the agent's status based on nectar and position
        if self.nectar == self.max_nectar:
            if self.pos == self.model.hive:
                self.status = 'u'  # Unloading at the hive
            else:
                self.status = 'c'  # Carrying nectar to the hive
        elif self.nectar > 0 and self.pos == self.model.hive:
            self.status = 'u'
        else:
            self.status = 's'  # Searching for nectar

    def step(self):
        self.update_status()

        if self.status == 'u':
            self.unload()

        if self.status == 'c':
            self.return_to_hive()

        if self.status == 's':
            self.search()



class FlowerAgent(mesa.Agent):
    def __init__(self, flower_agent_id, bee_model):
        super().__init__(flower_agent_id, bee_model)
        self.nectar = random.randint(0,100)


    def step(self):
        self.nectar += random.randint(0,2)



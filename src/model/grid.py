import mesa


class BeeGrid(mesa.space.MultiGrid):

    def __init__(self, size, resolution):
        super().__init__(size * resolution, size * resolution, False)
        self.hive = (size * resolution // 2, size * resolution // 2)
        self.dance_floor = (self.hive[0] + 1, self.hive[1])

class Edge:
    def __init__(self, speed_limit, distance, additional_costs=0):
        self.weight = (distance / speed_limit) + additional_costs
        self.speed_limit = speed_limit
        self.distance = distance
        self.additional_costs = additional_costs
        self.time_weight = (distance / speed_limit)

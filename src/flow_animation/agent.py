class Agent:

    destination = None
    origin = None
    space = None

    road_id: int = None
    location: float = None  # location on the road [0, 1]
    speed = None  # speed of the agent

    def __init__(self, origin, destination):
        self.destination = destination
        self.origin = origin

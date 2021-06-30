
def clamp(x, lo, hi):
    if x < lo:
        return lo, True
    elif x > hi:
        return hi, True
    return x, False

class x_controller:
    def __init__(self):
        self.x = 0
        self.x_dot = 0
        self.x_ddot = 0

        self.dt = 1/100
        self.rateLimit = 1/10

    def step(self, a):
        self.x_ddot, _ = clamp(a, self.x_ddot - self.rateLimit, self.x_ddot + self.rateLimit)
        self.x_dot += self.dt * self.x_ddot
        self.x += self.dt * self.x_dot
        
        self.x, _ = clamp(self.x, -1 , 1)

class Agent:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def select_action(self, obs, probabilistic=False):
        pass

    def update(self):
        raise NotImplementedError()

    def step(self, evaluate=False):
        raise NotImplementedError()

    def _to_tensordict(self, obs, action, reward, next_obs, done):
        pass

    def train(self, num_episodes):
        raise NotImplementedError()

    def evaluate(self):
        pass

    def save(self, file=None):
        raise NotImplementedError()

    def load(self, file=None):
        raise NotImplementedError()


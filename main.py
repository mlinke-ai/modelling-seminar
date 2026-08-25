import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import envs


def main():
    torch.autograd.set_detect_anomaly(True)
    env = envs.WallSimple()
    env.render_mode = None
    config = agents.SACAgentConfig()

    agent = agents.SACAgent(env, config)
    #agent.load()
    history = agent.train(num_episodes=10)
    #agent.save()

    env.close()


if __name__ == "__main__":
    main()

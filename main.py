import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import envs


def main():
    torch.autograd.set_detect_anomaly(True)
    #torch.cuda.memory._record_memory_history(max_entries=100000)
    env = envs.WallSimple()
    env.render_mode = None
    config = agents.SACAgentConfig()

    agent = agents.SACAgent(env, config)
    #agent.load()
    try:
        history = agent.train(num_episodes=5)
    except torch.OutOfMemoryError as e:
        print(e)
    finally:
        env.close()
        #torch.cuda.memory._dump_snapshot("profile.pkl")
        #torch.cuda.memory._record_memory_history(enabled=None)
    #agent.save()


if __name__ == "__main__":
    main()

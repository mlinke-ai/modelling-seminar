import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium

import agents
import envs

EPISODES = 500

def main():
    torch.autograd.set_detect_anomaly(True)
    #torch.cuda.memory._record_memory_history(max_entries=100000)

    #env = gymnasium.make("Pendulum-v1")
    env = envs.WallSimple()

    sac_config = agents.SACAgentConfig()
    sac_agent = agents.SACAgent(env, sac_config)
    try:
        sac_history = sac_agent.train(train_episodes=EPISODES)
    except torch.OutOfMemoryError as e:
        sac_history = dict(rewards=np.zeros(EPISODES, dtype=np.float32), steps=np.zeros(EPISODES, dtype=np.uint32))
        print(e)
    finally:
        env.reset()
        #torch.cuda.memory._dump_snapshot("profile.pkl")
        #torch.cuda.memory._record_memory_history(enabled=None)

    td3_config = agents.TD3AgentConfig()
    td3_agent = agents.TD3Agent(env, td3_config)
    try:
        td3_history = td3_agent.train(train_episodes=EPISODES)
    except torch.OutOfMemoryError as e:
        td3_history = dict(rewards=np.zeros(EPISODES, dtype=np.float32), steps=np.zeros(EPISODES, dtype=np.uint32))
        print(e)
    finally:
        env.reset()

    fig, (rewards, steps) = plt.subplots(2, 1, sharex=True)
    rewards.plot(sac_history["rewards"], label="SAC")
    rewards.plot(td3_history["rewards"], label="TD3")
    rewards.set_xlabel("Episodes")
    rewards.set_ylabel("Reward")
    rewards.legend()
    steps.plot(sac_history["steps"], label="SAC")
    steps.plot(td3_history["steps"], label="TD3")
    steps.set_xlabel("Episodes")
    steps.set_ylabel("Steps")
    steps.legend()
    fig.suptitle("Evaluation History")
    fig.savefig("plot.png")


if __name__ == "__main__":
    main()

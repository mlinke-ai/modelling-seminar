import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium

import agents
import envs

EPISODES = 5

def main():
    torch.autograd.set_detect_anomaly(True)
    #torch.cuda.memory._record_memory_history(max_entries=100000)

    env = gymnasium.make("Pendulum-v1")

    sac_config = agents.SACAgentConfig()
    sac_agent = agents.SACAgent(env, sac_config)
    try:
        sac_history = sac_agent.train(num_episodes=EPISODES)
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
        td3_history = td3_agent.train(num_episodes=EPISODES)
    except torch.OutOfMemoryError as e:
        td3_history = dict(rewards=np.zeros(EPISODES, dtype=np.float32), steps=np.zeros(EPISODES, dtype=np.uint32))
        print(e)
    finally:
        env.reset()

    """
    ppo_config = agents.PPOAgentConfig()
    ppo_agent = agents.PPOAgent(env, ppo_config)
    try:
        ppo_history = ppo_agent.train(num_episodes=EPISODES)
    except torch.OutOfMemoryError as e:
        ppo_history = dict(rewards=np.zeros(EPISODES, dtype=np.float32), steps=np.zeros(EPISODES, dtype=np.uint32))
        print(e)
    finally:
        env.reset()
    """

    plt.plot(sac_history["rewards"], label="SAC")
    plt.plot(td3_history["rewards"], label="TD3")
    #plt.plot(ppo_history["rewards"], label="PPO")
    plt.legend()
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("plot.png")


if __name__ == "__main__":
    main()

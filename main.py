import argparse

import gymnasium
import gymnasium.wrappers as wrappers
import matplotlib.pyplot as plt
import numpy as np
import torch

import agents
import envs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1, help="The number of episodes to train for.")
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    # torch.cuda.memory._record_memory_history(max_entries=100000)

    # env = gymnasium.make("Pendulum-v1")
    env = envs.WallSimple(render_mode="rgb_array")
    # env = wrappers.RecordVideo(env, video_folder="./recordings", episode_trigger=lambda t: (t + 1) % 100 == 0)

    sac_config = agents.SACAgentConfig()
    sac_agent = agents.SACAgent(env, sac_config)
    try:
        sac_history = sac_agent.train(train_episodes=args.episodes)
    except torch.OutOfMemoryError as e:
        sac_history = dict(rewards=np.zeros(EPISODES, dtype=np.float32), steps=np.zeros(EPISODES, dtype=np.uint32))
        print(e)
    finally:
        env.reset()
        # torch.cuda.memory._dump_snapshot("profile.pkl")
        # torch.cuda.memory._record_memory_history(enabled=None)

    """
    td3_config = agents.TD3AgentConfig()
    td3_agent = agents.TD3Agent(env, td3_config)
    try:
        td3_history = td3_agent.train(train_episodes=EPISODES)
    except torch.OutOfMemoryError as e:
        td3_history = dict(rewards=np.zeros(EPISODES, dtype=np.float32), steps=np.zeros(EPISODES, dtype=np.uint32))
        print(e)
    finally:
        env.reset()
    """

    fig, (rewards, steps) = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(15, 9))
    rewards[0].plot(sac_history["train_rewards"], label="SAC")
    # rewards[0].plot(td3_history["train_rewards"], label="TD3")
    rewards[0].set_xlabel("Training Episodes")
    rewards[0].set_ylabel("Training Reward")
    rewards[1].plot(sac_history["eval_rewards"], label="SAC")
    # rewards[1].plot(td3_hisotry["eval_rewards"], label="TD3")
    rewards[1].set_xlabel("Evaluation Episodes")
    rewards[1].set_ylabel("Evaluation Reward")
    steps[0].plot(sac_history["train_steps"], label="SAC")
    # steps[0].plot(td3_history["train_steps"], label="TD3")
    steps[0].set_xlabel("Training Episodes")
    steps[0].set_ylabel("Training Steps")
    steps[1].plot(sac_history["eval_steps"], label="SAC")
    # steps[1].plot(td3_hisotry["eval_steps"], label="TD3")
    steps[1].set_xlabel("Evaluation Episodes")
    steps[1].set_ylabel("Evaluation Steps")
    fig.legend(labels=["SAC"], loc="lower center")
    fig.suptitle("Training and Evaluation History")
    fig.savefig("plot.png")


if __name__ == "__main__":
    main()

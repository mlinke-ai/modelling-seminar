import argparse
import datetime
import pathlib

import gymnasium
import gymnasium.wrappers as wrappers
import numpy as np
import torch

import agents
import envs

AGENTS = {"td3": agents.TD3Agent, "sac": agents.SACAgent}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, choices=AGENTS.keys(), help="The agent to train.")
    parser.add_argument("episodes", type=int, help="The number of episodes to train for.")
    parser.add_argument("weights", type=pathlib.Path, help="The file to load and save model weights.")
    parser.add_argument("--record", type=int, default=0, help="When to record episodes.")
    parser.add_argument("--metrics", type=pathlib.Path, default=None, help="The file to store training and evaluation metrics into.")
    args = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    # torch.cuda.memory._record_memory_history(max_entries=100000)

    # env = gymnasium.make("Pendulum-v1")
    env = envs.WallSimple(render_mode="rgb_array", randomize_wall_pos=True)
    if args.record > 0:
        env = wrappers.RecordVideo(env, video_folder="./recordings", episode_trigger=lambda t: (t + 1) % args.record == 0)

    config = AGENTS[args.agent].get_default_config()
    agent = AGENTS[args.agent](env, config)
    if args.weights.exists():
        agent.load(args.weights)
    try:
        history = agent.train(train_episodes=args.episodes)
    except torch.OutOfMemoryError as e:
        print(e)
    else:
        agent.save(args.weights)
        if args.metrics is not None:
            if args.metrics.exists():
                metrics = np.load(args.metrics)
                for k, v in history.items():
                    history[k] = np.concat((metrics[k], v))
            np.savez(args.metrics, **history)
    finally:
        env.close()
        # torch.cuda.memory._dump_snapshot("profile.pkl")
        # torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()

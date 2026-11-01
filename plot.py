import argparse

import matplotlib.pyplot as plt

def main():
    fig, (rewards, steps) = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(15, 9))
    rewards[0].plot(history["train_rewards"], label="SAC")
    rewards[0].set_xlabel("Training Episodes")
    rewards[0].set_ylabel("Training Reward")
    rewards[1].plot(history["eval_rewards"], label="SAC")
    rewards[1].set_xlabel("Evaluation Episodes")
    rewards[1].set_ylabel("Evaluation Reward")
    steps[0].plot(history["train_steps"], label="SAC")
    steps[0].set_xlabel("Training Episodes")
    steps[0].set_ylabel("Training Steps")
    steps[1].plot(history["eval_steps"], label="SAC")
    steps[1].set_xlabel("Evaluation Episodes")
    steps[1].set_ylabel("Evaluation Steps")
    fig.legend(label=["SAC"], loc="lower center")
    fig.suptitle("Training and Evaluation History")
    fig.savefig("plot.png")

if __name__ == "__main__":
    main()


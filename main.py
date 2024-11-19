import gym
import numpy as np
from algorithms.Random import random_
from algorithms.SARSA import train_sarsa
from algorithms.QLearning import train_qlearning
from algorithms.DQN import train_dqn
from algorithms.DDQN import train_ddqn
from algorithms.Actor_Critic import train_actor_critic


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


algorithm = "ddqn"
episodes=500

env = gym.make('CartPole-v1',render_mode="human")

if algorithm == "random":
    random_(env)
elif algorithm == "sarsa":
    train_sarsa(env,episodes=episodes)
elif algorithm == "qlearning":
    train_qlearning(env,episodes=episodes)
elif algorithm == "dqn":
    train_dqn(env,episodes=episodes)
elif algorithm == "ddqn":
    train_ddqn(env,episodes=episodes)




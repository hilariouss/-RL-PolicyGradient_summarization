# Asynchronous Advantage Actor-critic (A3C) algorithm
# Paper link: https://arxiv.org/abs/1602.01783
# reference: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_continuous_action.py
#            https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/10_A3C/A3C_discrete_action.py
# Author: Dohyun, Kwon
# Date: 14th, Mar., 2019.
# Environment: OpenAI Gym, CartPole-v0
# Algorithm: (off-policy) A3C

# Synchronous and deterministic algorithm : A2C
# Multiple actors (and thus asynchronous)
# 다수의 worker의 parallel training

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--MAX_EPISODE', type=int, default=1000,
                    help='Max number of episode')
parser.add_argument('--GAME', default='CartPole-v0',
                    help='Game name')
parser.add_argument('--LOG_DIR', default='./log',
                    help='log directory')
parser.add_argument('--OUTPUT_GRAPH', default='True',
                    help='Tensorflow graph restoration')
parser.add_argument('--NUMBER_WORKERS', type=int, default=multiprocessing.cpu_count(),
                    help='The number of workers in parallel learning')
parser.add_argument('--GAMMA', type=float, default=0.99,
                    help='Discount factor')
parser.add_argument('--GLOBAL_NET_SCOPE', default='Global_Net',
                    help='Global network scope')
parser.add_argument('--LEARNING_RATE_ACTOR', type=float, default=0.001,
                    help='learning rate of actor')
parser.add_argument('--LEARNING_RATE_CRITIC', type=float, default=0.001,
                    help='learning rate of critic')
parser.add_argument('--ENTROPY_BETA', type=float, default=0.001,
                    help='Entropy beta')
parser.add_argument('--UPDATE_GLOBAL_ITER', type=int, default=10,
                    help='Period of updating global net')
hparams = parser.parse_args()

env = gym.make(hparams.GAME)
env.seed(1)
tf.set_random_seed(1)
np.random.seed(1)

n_state_space = env.observation_space.shape[0]
n_action_space = env.action_space.n

GLOBAL_EPISODE = 0
GLOBAL_REWARD = []





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
parser.add_argument('')
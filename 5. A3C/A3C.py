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
parser.add_argument('--GAME', default='CartPole-v0',
                    help='Game name')
parser.add_argument('--MAX_EPISODE', type=int, default=1000,
                    help='Max number of episode')
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
# In actor-critic, the td-error is used to calculate the loss of critic. e.g., tf.reduce_mean(tf.square(td_error)).
# On the other hand, the actor's loss is calculated by
class ACNet(object):
    def __init__(self, scope, globalAC=None):
        # Global network part
        if scope == hparams.GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(dtype=tf.float32, shape=[None, n_state_space], name="Global_state")
                self.actor_params, self.critic_params = self.build_networks(scope)[-2:]

        # Local network part
        else:
            with tf.variable_scope(scope):
                self.state = tf.placeholder(dtype=tf.float32, shape=[None, n_state_space], name="Local_state")
                self.action_history = tf.placeholder(dtype=tf.int32, shape=[None, ], name="Local_action") # index
                self.value_target = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Value_target") # why 1?

                self.pi_theta, self.value, self.actor_params, self.critic_params = self.build_networks(scope)

                td_error = tf.subtract(self.value_target, self.value, name="td_error")
                with tf.name_scope("critic_loss"):
                    self.critic_loss = tf.reduce_mean(tf.square(td_error))

                with tf.name_scope("actor_loss"):
                    log_pi_theta = tf.reduce_mean(tf.log(self.pi_theta + 1e+5) * tf.one_hot(self.action_history, n_action_space, dtype=tf.float32),
                                                                                            axis=1, keep_dims=True)
                    expected_value = log_pi_theta * tf.stop_gradient(td_error) #->expected_value에 음의부호넣어
                                                                               # actor의 loss로.
                    # gradient는 log_pi_theta에만 적용하기 때문에, td_error에는 tf.stop_gradient
                    # 음의 로그우도로 actor의 loss 계산.
                    # r + gamma*V(s') => target
                    # ---------------
                    #    ^
                    #    |
                    # target - V(s) = td-error (expectation of td-error is advantage --> Q(s, a) - V(s))
                    #                          (V ^ pi(s) = sigma_a[pi(a | s) * Q(s, a)] )
                    # baseline: V(s)
                    entropy = -tf.reduce_sum(self.pi_theta * tf.log(self.pi_theta + 1e-5), axis=1, keep_dims=True)
                    # cross entropy H(p, q) = - sigma_x [ p(x)log(q(x)) ] (x는 확률변수, p는 label, q는 logits)
                    # 모델의 예측 확률분포 (q)와 실제 확률분포 (p)의 오차가 작을수록 교차 엔트로피 H의 값이 작아지므로
                    # 이를 minimize하도록 학습하면 예측 확률분포 q가 실제 확률분포 p와 유사하게(오차가 작아지게)학습된다.
                    self.expected_value = hparams.ENTROPY_BETA * entropy + expected_value
                    self.actor_loss = tf.square(-self.expected_value)

                with tf.name_scope("local_gradient"):
                    self.actor_grads = tf.gradients(self.actor_loss, self.actor_params)
                    self.critic_grads = tf.gradients(self.critic_loss, self.critic_params)

            with tf.name_scope("sync"):
                with tf.name_scope('pull'): # local AC params <- globalAC params
                    self.pull_actor_params_op = [local_param.assign(global_param) for local_param, global_param in
                                                 zip(self.actor_params, globalAC.actor_params)]
                    self.pull_critic_params_op = [local_param.assign(global_param) for local_param, global_param in
                                                  zip(self.critic_params, globalAC.critic_params)]

                with tf.name_scope('push'):
                    self.update_actor_op = Actor_Optimizer.apply_gradients(zip(self.actor_grads, globalAC.actor_params))
                    self.update_critic_op = Critic_Optimizer.apply_gradients(zip(self.critic_grads, globalAC.critic_params))

    def build_networks(self, scope):
        weight_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            h_a = tf.layers.dense(inputs=self.state, units=200, activation=tf.nn.relu6,
                                  kernel_initializer=weight_init, bias_initializer=tf.constant_initializer(0.3),
                                  name='hidden_layer_actor')
            pi_theta = tf.layers.dense(inputs=h_a, units=n_action_space, activation=tf.nn.softmax,
                                       kernel_initializer=weight_init, bias_initializer=tf.constant_initializer(0.3),
                                       name='probability_actor')

        with tf.variable_scope('critic'):
            h_c = tf.layers.dense(inputs=self.state, units=100, activation=tf.nn.relu6,
                                  kernel_initializer=weight_init, bias_initializer=tf.constant_initializer(0.3),
                                  name='hidden_layer_critic')
            value = tf.layers.dense(inputs=h_c, units=1, activation=None,
                                    kernel_initializer=weight_init, bias_initializer=tf.constant_initializer(0.3),
                                    name='value_critic')

        # !!! findout tf.get_collection, tf.GraphKeys !!!
        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')
        return pi_theta, value, actor_params, critic_params

    def update_global_network(self, feed_dict): # run by each local net.
        sess.run([self.update_actor_op, self.update_critic_op], feed_dict=feed_dict)

    def pull_global(self): # run by each local net.
        sess.run([self.pull_actor_params_op, self.pull_critic_params_op])

    def choose_action(self, state):
        action_prob = sess.run(self.pi_theta, feed_dict={self.state: state[np.newaxis, :]})
        action = np.random.choice(range(action_prob.shape[1]),
                                  p=action_prob.ravel())

        return action

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(hparams.GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self):
        global GLOBAL_EPISODE, GLOBAL_REWARD
        total_step = 1
        buffer_state, buffer_action, buffer_reward = [], [], []
        while not Coordinator.should_stop() and GLOBAL_EPISODE < hparams.MAX_EPISODE: # !!! Coordinator search
            state = env.reset()
            episode_rwd = 0
            while True:
                # if self.name == 'Worker_0': self.env.render()
                action = self.AC.choose_action(state)
                state_, reward, done, info = self.env.step(action)
                if done:
                    reward -= 5
                episode_rwd += reward

                buffer_state.append(state)
                buffer_action.append(action)
                buffer_reward.append(reward)

                # Update the global and assign to local net
                if total_step % hparams.UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = sess.run(self.AC.value, feed_dict={self.AC.state: state_[np.newaxis, :]})[0, 0]

                    buffer_v_target = []
                    for r in buffer_reward[::-1]: # reverse buffer_reward
                        v_s_ = r + hparams.GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_state, buffer_action, buffer_v_target = np.vstack(buffer_state), np.array(buffer_action), np.vstack(buffer_v_target)

                    self.AC.update_global_network(feed_dict={self.AC.state: buffer_state,
                                                             self.AC.action_history: buffer_action,
                                                             self.AC.value_target: buffer_v_target})

                    buffer_state, buffer_action, buffer_reward = [], [], []
                    self.AC.pull_global()

                state = state_
                total_step += 1
                if done:
                    if len(GLOBAL_REWARD) == 0:
                        GLOBAL_REWARD.append(episode_rwd)
                    else:
                        GLOBAL_REWARD.append(0.99 * GLOBAL_REWARD[-1] + 0.01 * episode_rwd)
                    print(self.name, "Episode: ", GLOBAL_EPISODE,
                          "| episode reward: %i" % GLOBAL_REWARD[-1])
                    GLOBAL_EPISODE += 1
                    break

if __name__ == "__main__":
    sess = tf.Session()

    with tf.device("/cpu:0"):
        Actor_Optimizer = tf.train.RMSPropOptimizer(hparams.LEARNING_RATE_ACTOR, name="RMSProp_OP_Actor")
        Critic_Optimizer = tf.train.RMSPropOptimizer(hparams.LEARNING_RATE_CRITIC, name="RMSProp_OP_Critic")
        global_AC = ACNet(hparams.GLOBAL_NET_SCOPE) # scope: Global_Net

        workers = []

        for i in range(hparams.NUMBER_WORKERS):
            i_name = 'Worker_%i' % i # worker name
            workers.append(Worker(i_name, global_AC))

    Coordinator = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    if hparams.OUTPUT_GRAPH:
        if os.path.exists(hparams.LOG_DIR):
            shutil.rmtree(hparams.LOG_DIR)
        tf.summary.FileWriter(hparams.LOG_DIR, sess.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        thread = threading.Thread(target=job)
        thread.start()
        worker_threads.append(thread)
    Coordinator.join(worker_threads)

    plt.plot(np.arange(len(GLOBAL_REWARD)), GLOBAL_REWARD)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

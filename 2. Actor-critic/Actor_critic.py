import tensorflow as tf
import numpy as np
import gym

# Author: Dohyun, Kwon
# Date: 8th, Mar., 2019.
# Environment: OpenAI Gym, CartPole-v0
# Algorithm: (on-policy) Actor-critic

"""
# ===== Actor-critic ================================= #

    * Wrap-up *
    * GRADIENT *
    1. REINFORCE: utilize (NEGATIVE LOG PROBABILITY * DISCOUNTED SUM OF REWARDS) 
                  as loss(=gradient). Minimize the loss. 
                  Because the calculation of discounted sum of reward needs to roll out
                  an entire episode, the REINFORCE is called as Monte-Carlo policy gradient method.

                  [theta = theta + lr * {nabla_theta log(pi_theta)} * DiscountedSumofRewards(R)]
                                   <-------------------- 정책 theta 변화량 ---------------------->|
                                   e.g., neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=actions)
                                         loss = tf.reduce_mean(neg_log_prob * discounted_and_normalized_rewards_)
                                                                                               ^
                                         |-----------------------------------------------------|
                                         v
                                         (* REINFORCE use Return: r_t + gamma*r_t+1 + ... gamma^T-1 * r_T)
                                                                  because it utilize rewards occurred in an episode.

                                         self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.expected_v)

                  However, the REINFORCE needs to roll out entire episode and thus
                  the variance of gradient is quite dynamically changed, i.e., throughout the trajectory,
                  reward and action probability distribution is severely changed.

                  This property drives the agent with REINFORCE to learn unstable.


    2. Actor-critic: While REINFORCE rolls out an episode to calculate gradient (for the discounted sum of rewards),
                     'actor-critic' method utilize TD-error, which is contrast to the Monte-Carlo method.

                     Fundamentally, the actor-critic method consists of actor network and critic network.

                     1) The actor network takes an action given that the state of agent is input.
                     2) The critic critiques the action taken from the actor network.

                        First, the actor network takes an action
                        Next, the environment returns next state and reward.
                        Next, the critic calculates the TD-error and gradient of value function,
                        and updates its value function by policy gradient manner,

                            omega = omega + lr_critic * {nabla_omega(r + gamma*v(s') - v(s)) }
                                                                     |<--- TD-error(loss) --->|

                                    e.g., self.td_error = self.r + gamma * self.v_ - self.v
                                          self.loss = tf.square(self.td_error) 

                                          self.train_optimizer = tf.train.AdamOptimizer(lr).minimize(self.loss)

                        Next, the actor updates its policy as:


                            theta = theta + lr_actor * {nabla_theta(log(pi_theta)) * TD-error }
                                            |<--------------- theta의 변화량 ------------------>|

                                    e.g., log_prob = tf.log(self.pi_theta[0, self.a])
                                          self.expected_v = tf.reduce_mean(log_prob * self.td_error)                                     
                                          self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.expected_v)
                                                                                    (= maximize the expected_v)

# ==================================================== # 
"""
np.random.seed(1)  # for reproducible
tf.set_random_seed(1)  # as same

# ===== Hyperparameter ========== #
Output_graph = False  # Output
max_episode = 3000
max_timestep = 1000
gamma = 0.9
lr_actor = 0.001
lr_critic = 0.01

Display_reward_threshold = 200 # Render env. if total episode reward is greater than this value.
Render = False                 # Rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)

n_state_space = env.observation_space.shape[0]
n_action_space = env.action_space.n

# ==================================================== #
"""
Actor network definition
"""

class Actor(object):
    def __init__(self, sess, n_state_space, n_action_space, lr=0.001):
        self.sess = sess

        self.state = tf.placeholder(dtype=tf.float32, shape=[1, n_state_space], name="state")
        self.action = tf.placeholder(dtype=tf.int32, shape=None, name="action")
        self.TD_error = tf.placeholder(dtype=tf.float32, shape=None, name="TD_error")

        with tf.variable_scope("Actor"):
            layer1 = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='layer1'
            )
            self.pi_theta = tf.layers.dense(
                inputs=layer1,
                units=n_action_space,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='Action_prob_prediction'
            )
        with tf.variable_scope("expected_q"):
            log_prob = tf.log(self.pi_theta[0, self.action])
            self.expected_q = tf.reduce_mean(log_prob * self.TD_error) # calculate gradient of this

        with tf.variable_scope("train"):
            self.train_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(-self.expected_q)
                                                                          # -> maximize 'self.expected_q'
    def learn(self, state, action, TD_error):
        mod_state = state[np.newaxis,:] # because the shape of 'CartPole-v0' state is (4, ).
                                     # To configure the shape for calculating matrix multiplication,
                                     # the shape of state should be modified.
        feed_dict = {self.state: mod_state, self.action: action, self.TD_error: TD_error}
        _, expected_v = self.sess.run([self.train_optimizer, self.expected_q], feed_dict=feed_dict)

    def choose_stochastic_action(self, state):
        mod_state = state[np.newaxis, :] # (4, ) -> (1, 4)
        pi_theta = self.sess.run(self.pi_theta, feed_dict={self.state: mod_state})
        return np.random.choice(np.arange(pi_theta.shape[1]), p=pi_theta.ravel()) # pi_theta shape: (1, n_action_space)
                                                                                  # return a int (index of action)
class Critic(object):
    def __init__(self, sess, n_state_space, lr=0.01):
        self.sess = sess

        self.state = tf.placeholder(dtype=tf.float32, shape=[1, n_state_space], name="state")
        self.V_ = tf.placeholder(dtype=tf.float32, shape=[1, 1], name="next_v")
        self.R = tf.placeholder(dtype=tf.float32, shape=None, name="reward")

        with tf.variable_scope("Critic"):
            layer1 = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name="layer1"
            )

            self.V = tf.layers.dense(
                inputs=layer1,
                units=1, # number of output (v function value)
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name="V"
            )

        with tf.variable_scope("squared_TD_error"):
            self.TD_error = self.R + gamma*self.V_ - self.V
            self.loss = tf.square(self.TD_error)

        with tf.variable_scope("train"): # update
            self.train_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

    def learn(self, state, reward, state_):
        mod_state, mod_state_ = state[np.newaxis, :], state_[np.newaxis, :]
        V_ = self.sess.run(self.V, feed_dict={self.state: mod_state_})
        V = self.sess.run(self.V, feed_dict={self.state: mod_state})
        TD_error, _ = self.sess.run([self.TD_error, self.train_optimizer], feed_dict={self.state: mod_state,
                                                                                      self.R: reward,
                                                                                      self.V: V,
                                                                                      self.V_: V_
                                                                                      })
        return TD_error

# ==================================================== #
# Start the learning
sess = tf.Session()

actor = Actor(sess, n_state_space=n_state_space, n_action_space=n_action_space, lr=lr_actor)
critic = Critic(sess, n_state_space=n_state_space, lr=lr_critic)

sess.run(tf.global_variables_initializer())

if Output_graph:
    tf.summary.FileWriter("logs/", sess.graph)

for episode in range(max_episode):
    state = env.reset()
    time_step=0
    rewards = []
    while True:
        if Render: env.render()

        action = actor.choose_stochastic_action(state)
        state_, reward, done, info = env.step(action)

        if done: reward = -20

        rewards.append(reward)

        TD_error = critic.learn(state, reward, state_)
        actor.learn(state, action, TD_error)

        state = state_
        time_step += 1

        if done or time_step >= max_timestep:
            episode_reward_sum = sum(rewards)

            if 'running_reward' not in globals():
                running_reward = episode_reward_sum
            else:
                running_reward = running_reward * 0.95 + episode_reward_sum * 0.05

            if running_reward > Display_reward_threshold: Render = True
            print("episode: ", episode, " reward: ", int(running_reward))
            break
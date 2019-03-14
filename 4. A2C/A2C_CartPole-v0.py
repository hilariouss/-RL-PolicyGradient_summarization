import tensorflow as tf
import numpy as np
import gym

# Reference ============================================= #
# https://github.com/sarcturus00/Tidy-Reinforcement-learning/blob/master/Algorithm/A2C_cartpole.py
# https://github.com/hilariouss/-RL-PolicyGradient_summarization/blob/master/2.%20Actor-critic/Actor_critic.py

np.random.seed(1)  # for reproducible
tf.set_random_seed(1)  # as same

# ===== Hyperparameter ========== #
Output_graph = False  # Output
max_episode = 1000
max_timestep = 100
gamma = 0.9
lr_actor = 0.001
lr_critic = 0.01

Display_reward_threshold = 200 # Render env. if total episode reward is greater than this value.
Render = False                 # Rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)

n_obs_space = env.observation_space.shape[0]
n_action_space = env.action_space.n

# ==================================================== #
# Actor network definition

class Actor(object):
    def __init__(self, action_space, name):
        self.action_space = action_space
        self.name = name

    def actor_network(self, observation, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(inputs=observation, units=10, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name='hidden_layer_1')
            logits_action_weights = tf.layers.dense(inputs=h1, units=self.action_space, activation=None,
                                       kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                       bias_initializer=tf.constant_initializer(0.1),
                                       name="pi_theta")
            return logits_action_weights
    # action probability distribution
    def get_action_prob(self, observation, reuse=False):
        logits_action_weights = self.actor_network(observation, reuse)
        action_prob = tf.nn.softmax(logits_action_weights, name='logits_action_weights') # what if change
        return action_prob
    """
    # return negative log likelihood (cross entropy)
    """
    def get_cross_entropy(self, observation, label_action_weights, reuse=True):
        logits_action_weights = self.actor_network(observation, reuse) # action probability distribution get.
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_action_weights,
                                                                       labels=label_action_weights)
        return cross_entropy

# ==================================================== #
# Critic network definition

class Critic(object):
    def __init__(self, name):
        self.name = name

    def critic_network(self, observation, reuse):
        with tf.variable_scope(self.name, reuse=reuse):
            h1 = tf.layers.dense(inputs=observation, units=10, activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 name="hidden_layer_1")
            value = tf.layers.dense(inputs=h1, units=1, activation=None,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    name="value")
            return value

    def get_value(self, observation, reuse=False):
        value = self.critic_network(observation=observation, reuse=reuse)
        return value

# ==================================================== #
# Replay memory

class ReplayMemory(object):
    def __init__(self):
        self.episode_observation, self.episode_action, self.episode_reward = [], [], []

    def store_transition(self, observation, action, reward):
        self.episode_observation.append(observation)
        self.episode_action.append(action)
        self.episode_reward.append(reward)

    def convert2array(self):
        array_obs = np.vstack(self.episode_observation)
        array_act = np.array(self.episode_action)
        array_rwd = np.array(self.episode_reward)
        return array_obs, array_act, array_rwd

    def reset(self):
        self.episode_observation, self.episode_action, self.episode_reward = [], [], []

# ==================================================== #
# A2C implementation

class A2C_Agent:
    def __init__(self, action_space, observation_space, lr_actor, lr_critic, gamma):
        # Parameter setting
        self.action_space = action_space
        self.observation_space = observation_space
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma

        # Placeholder: state, action, q value
        self.observation = tf.placeholder(dtype=tf.float32, shape=[None, self.observation_space], name="observation")
        self.action_label = tf.placeholder(dtype=tf.int32, shape=[None], name="action")  # label
        self.Q = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="Q_value")

        # Instance: actor/critic/replay memory initialization
        actor = Actor(self.action_space, 'actor')
        critic = Critic('critic')
        self.memory = ReplayMemory()

        # Actor network - calculate 1) logits of action weights and 2) cross entropy(loss of actor)
        self.action_prob = actor.get_action_prob(observation=self.observation)
        cross_entropy = actor.get_cross_entropy(observation=self.observation, label_action_weights=self.action_label)
        # Critic network - calculate 1) value
        self.value = critic.get_value(observation=self.observation)

        # calculate the advantage
        self.advantage = self.Q - self.value

        # loss of actor and critic
        actor_loss = tf.reduce_mean(cross_entropy * self.advantage)
        self.actor_train_optimizer = tf.train.AdamOptimizer(learning_rate=lr_actor).minimize(actor_loss)
        critic_loss = tf.reduce_mean(tf.square(self.advantage))
        self.critic_train_optimizer = tf.train.AdamOptimizer(learning_rate=lr_critic).minimize(critic_loss)

        # agent's session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    # Note that all episodic reward is not available not as REINFORCE. (because of TD manner)
    def compute_Q(self, last_value, done, reward):
        Q = np.zeros_like(reward)
        if done:
            value = 0
        else:
            value = last_value

        for t in reversed(range(0, len(reward))):
            value = value * self.gamma + reward[t]
            Q[t] = value
        return Q[:, np.newaxis]

    def step(self, observation): # observation : env.reset()
        if observation.ndim < 2: observation = observation[np.newaxis, :]
        action_prob = self.sess.run(self.action_prob, feed_dict={self.observation: observation}) ###
        action = np.random.choice(range(action_prob.shape[1]), p=action_prob.ravel()) # value of weight selected from prob. distribution 'p'.
        value = self.sess.run(self.value, feed_dict={self.observation: observation})
        return action, value

    # optimizer
    def learn(self, last_value, done):
        observation, action, reward = self.memory.convert2array()
        Q = self.compute_Q(last_value, done, reward)

        self.sess.run(self.actor_train_optimizer, feed_dict={self.observation: observation, self.action_label: action,
                                                             self.Q: Q})
        self.sess.run(self.critic_train_optimizer, feed_dict={self.observation: observation, self.Q: Q})

        self.memory.reset()

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

agent = A2C_Agent(action_space=env.action_space.n, observation_space=env.observation_space.shape[0],
                             lr_actor=0.01, lr_critic=0.02, gamma=0.99)

total_episode = 1000
max_step = 200

for epi in range(total_episode):
    obs = env.reset()
    episode_rwd = 0

    while True:
        act, _ = agent.step(obs)
        obs_, rwd, done, info = env.step(act)

        agent.memory.store_transition(obs, act, rwd)
        episode_rwd += rwd

        obs = obs_

        if done:
            _, last_value = agent.step(obs_)
            agent.learn(last_value, done)
            break

    print("Episode: %i" % epi, "|Episode reward: %i" % episode_rwd)

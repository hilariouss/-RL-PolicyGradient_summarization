import numpy as np
import tensorflow as tf
import gym

# reference: https://fakabbir.github.io/reinforcement-learning/docs/cartpole/
#            https://github.com/hilariouss/-RL-PolicyGradient_summarization/tree/master/1.%20REINFORCE/Reference/REINFORCE
# Author: Dohyun, Kwon
# Date: 8th, Mar., 2019.
# Environment: OpenAI Gym, CartPole-v0
# Algorithm: (on-policy) REINFORCE

env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1) # Since PG has high variance, static seed value for replay

# ===== 1. Hyperparameter ========== #
# == 1-1. Environment parameter == #
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# == 1-2. Learning parameter == #
total_episodes = 1000
learning_rate = 1e-2
gamma = 0.95
#=================================#


# ===== 2. cumulative reward function #
def discounted_and_normalized_rewards(episode_rewards):
    # Input : episode_rewards --> list of rewards in an episode.
    # Return : discounted and normalized rewards list
    discounted_and_normalized_rewards = np.zeros_like(episode_rewards)
    cumulative_sum = 0.0 # for future reward value

    for i in reversed(range(len(episode_rewards))):
        cumulative_sum = cumulative_sum * gamma + episode_rewards[i]
        discounted_and_normalized_rewards[i] = cumulative_sum
        # Firstly, allocate the cumulative sum of reward to corresponding reward among trajectory(state)

    mean = np.mean(discounted_and_normalized_rewards)
    std = np.std(discounted_and_normalized_rewards)
    discounted_and_normalized_rewards = (discounted_and_normalized_rewards - mean) / std # Normalization

    return discounted_and_normalized_rewards


# ===== 3. Create policy gradient with neural network model ========== #
with tf.name_scope('inputs'):
    input_ = tf.placeholder(dtype=tf.float32, shape=[None, state_size], name="input_")
    actions = tf.placeholder(dtype=tf.int32, shape=[None, action_size], name="actions")
    discounted_and_normalized_rewards_ = tf.placeholder(dtype=tf.float32, shape=[None, ], name="discounted_episode_rewards")

    mean_reward_ = tf.placeholder(dtype=tf.float32, name="mean_reward") # Add this placeholder to show in tensorboard

    with tf.name_scope('fc1'):
        fc1 = tf.contrib.layers.fully_connected(inputs=input_,
                                                num_outputs=10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope('fc2'):
        fc2 = tf.contrib.layers.fully_connected(inputs=fc1,
                                                num_outputs=action_size,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope('prob'):
        softmax_action_prob_distribution = tf.contrib.layers.fully_connected(inputs=fc2,
                                                                             num_outputs=action_size,
                                                                             activation_fn=tf.nn.softmax,
                                                                             weights_initializer=tf.contrib.layers.xavier_initializer())


    with tf.name_scope('loss'):
        # Compute the loss via calculating CROSS ENTROPY
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax_action_prob_distribution, labels=actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_and_normalized_rewards_)

    with tf.name_scope('train'):
        train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# ===== 4. Set up the tensorboard ========== #
Writer = tf.summary.FileWriter("/tensorboard/PolicyGradient/tensorboard_result")

# losses
tf.summary.scalar("Loss", loss)
# Reward mean
tf.summary.scalar("Reward mean", mean_reward_)

Write_op = tf.summary.merge_all()

# ===== 5. Train the agent ========== #
Episodic_rewards = [] # Rewards sum of each episode
Total_reward = 0 # Rewards sum over all episodes
Maximum_recorded_reward = 0 # Maximum episode reward among all episodes
episode = 0
episode_states, episode_actions, episode_rewards = [], [], []

# model saver
saver = tf.train.Saver()

"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(total_episodes):
        episode_rewards_sum = 0 # will be appended as element of 'Episodic_rewards'
        # Launch the game
        state = env.reset()

        env.render()

        while True:
            # Choose an action with stochastic manner.

            # get action distribution (prediction, shape: [1, 4])
            action_prob_distribution = sess.run(softmax_action_prob_distribution, feed_dict={input_: state.reshape([1, 4])})
            # pick action with the action probability distribution
            action_index = np.random.choice(range(action_prob_distribution.shape[1]), p=action_prob_distribution.ravel())

            # perform the action
            state_, reward, done, info = env.step(action_index)

            # Store the s, a, r
            episode_states.append(state)

            # Mark the picked action as 1 for corresponding action_index
            action_performed = np.zeros(action_size)
            action_performed[action_index] = 1

            # Store the action list and reward
            episode_actions.append(action_performed)
            episode_rewards.append(reward)

            if done:
                # Calculate sum reward
                episode_rewards_sum = np.sum(episode_rewards) # Rewards sum within an episode
                Episodic_rewards.append(episode_rewards_sum)  # Append current episodic reward
                Total_reward = np.sum(Episodic_rewards)       # Sum total episodes reward

                # Mean reward
                mean_reward = np.divide(Total_reward, episode +1)

                # maximum reward
                Maximum_recorded_reward = np.amax(Episodic_rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean reward among episodes: ", mean_reward)
                print("Max reward so far: ", Maximum_recorded_reward)

                # Calculate discounted reward
                discounted_episode_rewards = discounted_and_normalized_rewards(episode_rewards)

                # Feedforward, gradient and backpropagation
                loss_, _ = sess.run([loss, train_optimizer], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                        actions: np.vstack(np.array(episode_actions)),
                                                                        discounted_and_normalized_rewards_: discounted_episode_rewards})

                # Write Tensorflow summaries
                summary = sess.run(Write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
                                                        actions: np.vstack(np.array(episode_actions)),
                                                        discounted_and_normalized_rewards_: discounted_episode_rewards,
                                                        mean_reward_: mean_reward})

                Writer.add_summary(summary, episode)
                Writer.flush()

                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [], [], []
                break
            state = state_
        # Save model
        if episode % 100 == 0:
            saver.save(sess, "./models/model.ckpt")
            print("Model is saved (episode: {})".format(episode))
"""
# ===== 6. Test the agent ========== #
with tf.Session() as sess:
    env.reset()
    rewards = []

    # Load the model
    saver.restore(sess, "./models/model.ckpt")

    for episode in range(10):
        state = env.reset()
        time_step = 0
        done = False
        total_reward = 0
        print("*"* 20)
        print("Episode {}".format(episode))

        while True:
            # choose action among action distribution
            action_prob_distribution = sess.run(softmax_action_prob_distribution, feed_dict={input_: state.reshape([1, 4])})
            action = np.random.choice(range(action_prob_distribution.shape[1]), p=action_prob_distribution.ravel())

            state_, reward, done, _ = env.step(action)

            total_reward += reward

            if done:
                rewards.append(total_reward)
                print("Score: ", total_reward)
                break
            state = state_

    env.close()
    print("Score over time: " + str(sum(rewards)/10))

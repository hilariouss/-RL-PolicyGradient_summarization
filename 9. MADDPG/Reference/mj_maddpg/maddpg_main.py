import numpy as np
import tensorflow as tf
from collections import deque
from maddpg import MADDPGAgent
from mlagents.envs import UnityEnvironment
from noise import OU
import random

#######################################

state_size = 39
action_size = 3

load_model = True
train_mode = True

batch_size = 128
mem_maxlen = 10000
discount_factor = 0.99
learning_rate = 0.001

run_episode = 200000
test_episode = 10
start_train_episode = 10
target_update_step = 5

save_interval = 10

epsilon = 0.756
epsilon_min = 0.1
softlambda = 0.9
env_name = "C:/Users/asdfw/Desktop/twoleg/twoleg"
save_path = "./maddpg_models/twoleg/"
load_path = "./maddpg_models/twoleg/"


class Normalizer():  # refer page 7 section 3.2 of paper
    def __init__(self, nb_inputs):  # nb_inputs is no. of perceptrons
        # initializing the variables required for normalization
        self.n = np.zeros(nb_inputs)  # total number of states. vector of zeros equal to no. of perceptrons
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):  # x is new state. function observe is called everytime we observe a new state
        self.n += 1.  # to make it float
        last_mean = self.mean.copy()  # saving mean before updating it
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(
            min=1e-2)  # clip(min = 1e-2) is used to make sure self.var is never equal to zero

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)  # std. deviation
        return (inputs - obs_mean) / obs_std  # normalized values


def get_agents_action(state1, state2, sess):
    agent1_action = agent1_ddpg.action(state=np.reshape(state1, newshape=[-1, state_size]), train_mode=train_mode, sess=sess)
    agent2_action = agent2_ddpg.action(state=np.reshape(state2, newshape=[-1, state_size]), train_mode=train_mode, sess=sess)
    #agent3_action = agent3_ddpg.action(state=np.reshape(state3, newshape=[-1, state_size]), train_mode=train_mode, sess=sess)
    #agent4_action = agent4_ddpg.action(state=np.reshape(state4, newshape=[-1, state_size]), train_mode=train_mode, sess=sess)
    return agent1_action, agent2_action

if __name__ == "__main__":

    # Environment Setting =================================
    env = UnityEnvironment(file_name=env_name, worker_id=2)

    # Set the brain for each players ======================
    brain_name1 = env.brain_names[1]
    brain_name2 = env.brain_names[2]
    #brain_name3 = env.brain_names[3]
    #brain_name4 = env.brain_names[4]

    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]
    #brain3 = env.brains[brain_name3]
    #brain4 = env.brains[brain_name4]

    step = 0

    rewards1 = []
    losses1 = []
    rewards2 = []
    losses2 = []
    #rewards3 = []
    #losses3 = []
    #rewards4 = []
    #losses4 = []

    agent_num = 2

    # Agent Generation =======================================
    agent1_ddpg = MADDPGAgent(agent_num, state_size, action_size, '1')
    agent2_ddpg = MADDPGAgent(agent_num, state_size, action_size, '2')
    #agent3_ddpg = MADDPGAgent(agent_num, state_size, action_size, '3')
    #agent4_ddpg = MADDPGAgent(agent_num, state_size, action_size, '4')

    # Save & Load ============================================
    Saver = tf.train.Saver(max_to_keep=5)
    load_path = load_path
    # self.Summary,self.Merge = self.make_Summary()
    # ========================================================

    # Session Initialize =====================================
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)

    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(load_path)
        Saver.restore(sess, ckpt.model_checkpoint_path)
        print("[Restore Model]")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("[Initialize Model]")

    # TargetNework 초기화 ====================================
    agent1_ddpg.target_init(sess)
    agent2_ddpg.target_init(sess)
    #agent3_ddpg.target_init(sess)
    #agent4_ddpg.target_init(sess)
    print("[Target Networks Initialized]")
    # ========================================================


    # Tensorboard ============================================
    reward_history = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    reward_op = [tf.summary.scalar('Agent_' + str(i) + '_reward', reward_history[i]) for i in range(agent_num)]
    summary_writer = tf.summary.FileWriter('./three_summary', graph=tf.get_default_graph())
    print("Tensorbard Initialized")
    # ========================================================

    # Reset Environment =======================
    env_info = env.reset(train_mode=train_mode)
    print("[Env Reset]")
    # =========================================

    # ====================================================
    # Noise parameters - Ornstein Uhlenbeck
    DELTA = 0.1  # The rate of change (time)
    SIGMA = 0.3 # Volatility of the stochastic processes
    OU_A = 15.  # The rate of mean reversion
    OU_MU = 0.  # The long run average interest rate
    # ====================================================

    noise = OU(DELTA, SIGMA, OU_A, OU_MU, 3)
    ou_level = 0.

    normalizer1 = Normalizer(state_size)
    normalizer2 = Normalizer(state_size)

    for episode in range(run_episode + test_episode):
        #if episode > 0 and episode % 10 == 0 :
        #    train_mode = False
        #else:
        #    train_mode = True

        env_info = env.reset(train_mode=train_mode)
        done = False

        # Brain Set ====================================
        state1 = env_info[brain_name1].vector_observations[0]
        episode_rewards1 = 0
        done1 = False

        state2 =  env_info[brain_name2].vector_observations[0]
        episode_rewards2 = 0
        done2 = False

        #state3 =  env_info[brain_name3].vector_observations[0]
        #episode_rewards3 = 0
        #done3 = False

        #state4 =  env_info[brain_name4].vector_observations[0]
        #episode_rewards4 = 0
        #done4 = False
        # =============================================
        step = 0
        dist = 0

        while not done:
            step += 1
            normalizer1.observe(state1)
            normalizer2.observe(state2)
            agent1_action, agent2_action = get_agents_action(normalizer1.normalize(state1), normalizer2.normalize(state2), sess)

            # e-greedy ======================================
            if train_mode == True:
                if episode % 10 == 0:
                    noise = OU(DELTA, SIGMA, OU_A, OU_MU, 3)
                    ou_level = 0.


                ou_level = noise.ornstein_uhlenbeck_level(ou_level) * 2
                action1 = agent1_action[0] + ou_level
                ou_level = noise.ornstein_uhlenbeck_level(ou_level) * 2
                action2 = agent2_action[0] + ou_level
            else:
                action1 = agent1_action[0]
                action2 = agent2_action[0]

            env_info = env.step(vector_action = {brain_name1: [action1], brain_name2: [action2]})

            next_state1 = env_info[brain_name1].vector_observations[0]
            reward1 = env_info[brain_name1].rewards[0]
            episode_rewards1 += reward1
            done1 = env_info[brain_name1].local_done[0]

            next_state2 = env_info[brain_name2].vector_observations[0]
            reward2 = env_info[brain_name2].rewards[0]
            episode_rewards2 += reward2
            done2 = env_info[brain_name2].local_done[0]

            #next_state3 = env_info[brain_name3].vector_observations[0]
            #reward3 = env_info[brain_name3].rewards[0]
            #episode_rewards3 += reward3
            #done3 = env_info[brain_name3].local_done[0]

            #next_state4 = env_info[brain_name4].vector_observations[0]
            #reward4 = env_info[brain_name4].rewards[0]
            #episode_rewards4 += reward4
            #done4 = env_info[brain_name4].local_done[0]

            done = done1 or done2

            # Memory Set ==============================
            if train_mode:
                data1 = [state1, agent1_action[0], agent2_action[0], reward1,
                         next_state1, next_state2, done1]
                data2 = [state2, agent2_action[0], agent1_action[0], reward2,
                         next_state2, next_state1, done2]

                agent1_ddpg.append_sample(data1)
                agent2_ddpg.append_sample(data2)

                if episode > start_train_episode and len(agent1_ddpg.memory) > 4 * batch_size:
                    for t in range(0,4):
                        loss1 = agent1_ddpg.train_models(agent2_ddpg.target_actor, normalizer1, normalizer2, sess)
                        loss2 = agent2_ddpg.train_models(agent1_ddpg.target_actor, normalizer2, normalizer1, sess)

                        losses1.append(loss1)
                        losses2.append(loss2)
            # =========================================

            state1 = next_state1
            state2 = next_state2
            #state3 = next_state3
            #state4 = next_state4

            if step > 500:
                break


        if episode > 0 and episode % 15 == 0:
            agent1_ddpg.target_update(sess)
            agent2_ddpg.target_update(sess)
            #agent3_ddpg.target_update(sess)
            #agent4_ddpg.target_update(sess)

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)
        #rewards3.append(episode_rewards3)
        #rewards4.append(episode_rewards4)

        if episode % 10 == 0 and episode != 0:
            print(
                "episode: {} / reward1: {:.2f} / reward2: {:.2f} / epsilon: {:.3f} / memory_len:{}".format
                (episode, np.mean(rewards1), np.mean(rewards2), epsilon, len(agent1_ddpg.memory)))
            print(
                "loss1: {:.2f} / loss2: {:.2f} ".format
                (np.mean(losses1), np.mean(losses2)))
            rewards1 = []
            rewards2 = []
            #rewards3 = []
            #rewards4 = []
            losses1 = []
            losses2 = []
            #losses3 = []
            #losses4 = []

        if episode % save_interval == 0:
            Saver.save(sess, save_path + "model.ckpt")

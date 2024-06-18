import matplotlib.pyplot as plt
import streamlit as sl
from keras.datasets import mnist
import numpy as np
import gym
from gym import spaces

from module.mnist_env import MNISTEnv

class q_learning():
    def __init__(self, Q=None, alpha=0.1, gamma=0.9, epsilon=1, epsilon_min=0.1, epsilon_decay=0.995, num_episode=32):                
        # Q-learning parameters
        self.Q = Q
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.num_episodes = num_episode

    def train(self, iterations=10000):

        env = MNISTEnv(type='train')

        # Initialize Q-table
        Q = np.zeros((env.h, env.w, 40))  # State-action space
        rewards = []    
        mean_reward_100 = []
        mean_reward_32 = []
        reward_32 = []
        for episode in range(self.num_episodes*iterations):
            # print each 100 iterations (100*32)
            display = True if episode%3200 == 0 else False
            
            save = True if episode%32 == 0 else False
            # print 
            state = env.reset()
            total_reward = 0
            
            done = False
            while not done:
                
                h, w = env.pos
                if np.random.rand() < self.epsilon:
                    action = env.action_space.sample()
                else:
                    action = np.argmax(Q[h, w])
                
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                
                next_h, next_w = env.pos
                best_next_action = np.argmax(Q[next_h, next_w])
                Q[h, w, action] = Q[h, w, action] + self.alpha * (reward + self.gamma * Q[next_h, next_w, best_next_action] - Q[h, w, action])
                
                state = next_state
            
            mean_reward_100.append(total_reward)
            mean_reward_32.append(total_reward)
            
            if save:
                if episode == 0:
                    reward
            
            if display:
                if episode==0:
                    print(f"Episode 0, total mean reward: 0")
                    rewards.append(0)
                else:
                    current_mean_100_reward = np.mean(np.array(mean_reward_100))
                    print(f"From episode {int((episode/32))-99} to {int(episode/32)}, total mean reward: {current_mean_100_reward}")
                    rewards.append(current_mean_100_reward)
                    mean_reward_100 = []
                
            epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        self.Q = Q    
        print("Training finished.")
        
        return rewards
        


    def eval(self):
        env = MNISTEnv(type='test')
        
        # Evaluation
        num_test_episodes = 1000
        reward_total = 0
        success_count = 0
        for _ in range(num_test_episodes):
            state = env.reset()
            done = False
            while not done:
                h, w = env.pos
                action = np.argmax(self.Q[h, w])
                state, reward, done, _ = env.step(action)
                reward_total += reward
                if done and reward > 0:
                    success_count += 1

        print(f"Average reward on 1000 test iterations:{reward_total/1000}")
        print(f"Success rate: {success_count / num_test_episodes * 100}%")
        
    def test(self):
        env = MNISTEnv(type='test', seed=7)
        state = env.reset()
        reward_total = 0
        step = 0
        
        done = False
        while not done:
            step += 1
            h, w = env.pos
            action = np.argmax(self.Q[h, w])
            state, reward, done, _ = env.step(action)
            sl.write('Action reward:', reward)
            env.render()
            
            reward_total += reward
            
            if step==20: break

        sl.write(f"Total reward:{reward_total}")
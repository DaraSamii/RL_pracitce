from collections import namedtuple
import math
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_dim, layer1_dim, layer2_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(in_features = input_dim, out_features = layer1_dim)
        self.fc2 = nn.Linear(in_features = layer1_dim, out_features =layer2_dim)
        self.out = nn.Linear(in_features = layer2_dim, out_features = output_dim)

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.Experience = namedtuple('Experience',
                                     ('state', 'action', 'reward','next_state'))


    def push(self, state, action, reward, next_state):
        e = self.Experience(state,action,reward,next_state)

        if len(self.memory) < self.capacity:
            self.memory.append(e)

        else:
            self.memory[self.push_count % self.capacity] = e

        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)


class Agent:
    def __init__(self,policy_net):
        self.policy_net = policy_net

    def sample(self,state):
        return random.randint(0,1)

    def policy_action(self,state):
        with torch.no_grad():
            print('heello')
            print(state)
            a = self.policy_net(state.float()).argmax().item()
            #print('l :',a)
            return a

class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        print(states)
        print('actions',actions.unsqueeze(-1))
       # print('policy_net-states', policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1)))
        return policy_net(states.float())#.gather(dim = 1, index = actions.unsqueeze(-1))

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.flatten(start_dim = 1) \
            .max(dim = 1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        print(target_net(non_final_states))
        values[non_final_state_locations] = target_net(non_final_states).max()[0].detach()
        return values


class learn:
    def __init__(self, capacity, Agent, batch_size, lr, gamma):
        self.memory = ReplayMemory(capacity)
        self.Agent = Agent
        self.target_net = self.Agent.policy_net
        self.target_net.load_state_dict(self.Agent.policy_net.state_dict())
        self.target_net.eval()
        self.batch_size = batch_size
        self.lr =lr
        self.gamma = gamma
        self.optimizer = optim.Adam(params = self.Agent.policy_net.parameters(), lr = lr)


    def train(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

        if self.memory.can_provide_sample(self.batch_size):
            experiences = self.memory.sample(self.batch_size)

            states = torch.cat([x.state for x in experiences])
            print(states)
            actions = torch.tensor([x.action for x in experiences])
            rewards = torch.tensor([x.reward for x in experiences])
            next_states = torch.cat([x.next_state for x in experiences])


            current_q_values = QValues.get_current(self.Agent.policy_net, states, actions)
            next_q_values = QValues.get_next(self.target_net, next_states)
            target_q_values = (next_q_values * self.gamma) + rewards

            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


import gym

env = gym.make('CartPole-v0')
env.reset()

policy_net = DQN(20,30,30,2)
agent = Agent(policy_net)

L = learn(10,agent,5,0.01,0.99)
strategy = EpsilonGreedyStrategy(1,0.01,0.001)

rewards = []
for episode in range(1000):
    state = torch.tensor(env.reset())

    eps_reward = 0
    if strategy.get_exploration_rate(episode)  > random.random():
        action = agent.sample(None)
    else:
        action = agent.policy_action(state)
    print('act', action)
    next_state, reward, done, _ = env.step(0)
    eps_reward =+ reward
    L.train(state,action,reward,torch.tensor(next_state))
    state = next_state

    if done == True:
        rewards.append(eps_reward)
        print(eps_reward)

        break



env.close()
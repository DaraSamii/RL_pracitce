import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import numpy as np


class DeepQnetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQnetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.lr = lr
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, state):
        x = self.fc1(state.float())
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(x,0.1)

        x = self.fc2(x)
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(x,0.1)
        
        actions = self.fc3(x)

        return actions


class Agent():
    def __init__(self, gamma, lr, input_dims, batch_size,
                 n_actions, max_mem_size = 100000, DQN_path = None,
                 replay_memory_path = None):

        self.gamma = gamma
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.replay_memory_path = replay_memory_path
        self.DQN_path = DQN_path
        self.action_space = [i for i in range(n_actions)]
        self.mem_cntr = 0

        self.Q_eval = DeepQnetwork(self.lr, n_actions = n_actions,
                                       input_dims = input_dims,
                                       fc1_dims = 256, fc2_dims = 256)

        if self.DQN_path != None:
            self.Q_eval.state_dict(T.load(self.DQN_path))

        if self.replay_memory_path == None:
            self.state_memory = np.zeros((self.mem_size, *input_dims),dtype = np.float32)
            self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
            self.action_memory = np.zeros(self.mem_size, dtype = np.float32)
            self.reward_memory = np.zeros(self.mem_size, dtype = np.bool)
            self.terminal_memory = np.zeros(self.mem_size, dtype= np.bool)

        else:
            file = open(self.replay_memory_path,'rb')
            replay_memory = pickle.load(file)
            self.state_memory = replay_memory['state_memory']
            self.new_state_memory = replay_memory['new_state_memory']
            self.action_memory = replay_memory['action_memory']
            self.reward_memory = replay_memory['reward_memory']
            self.terminal_memory = replay_memory['terminal_memory']
            self.mem_cntr = replay_memory['mem_cntr']
            file.close()

    def save(self,DQN_path, Memory_path):
        T.save(self.Q_eval.state_dict(),DQN_path)

        memory_dic = {}
        memory_dic['state_memory'] = self.state_memory
        memory_dic['new_state_memory'] = self.new_state_memory
        memory_dic['action_memory'] = self.action_memory
        memory_dic['reward_memory'] = self.reward_memory
        memory_dic['terminal_memory'] = self.terminal_memory
        memory_dic['mem_cntr'] = self.mem_cntr

        file = open(Memory_path,'wb')
        pickle.dump(memory_dic, file)
        file.close()


    def store_tranisitions(self, state, action,reward, state_, done):
        index= self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation, epsilon):
        if np.random.random() > epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()

        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem,self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward((state_batch))[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)

        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        loss= self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)

        loss.backward()

        self.Q_eval.optimizer.step()












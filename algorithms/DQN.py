import torch
import torch.nn as nn
import torch.optim as optim
import gym
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def train_dqn(env, episodes, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
              learning_rate=1e-3, batch_size=32, buffer_size=1000000, target_update_freq=10):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device:{device}')

    policy_net = DQN(input_dim, output_dim).to(device)
    target_net = DQN(input_dim, output_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    memory = deque(maxlen=buffer_size)

    episode_rewards = []  # 存储每个 episode 的总奖励
    moving_avg_rewards = []  # 存储滑动平均奖励

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()  # 探索：随机选择动作
            else:
                state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)  # 将状态转为 Tensor
                action = torch.argmax(policy_net(state_tensor)).item()  # 利用：选择 Q 值最大的动作

            next_state, reward, done, truncated, _ = env.step(action)

            memory.append((state, action, reward, next_state, done))

            # 如果经验池的大小大于等于批量大小，则进行学习
            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states_tensor = torch.FloatTensor(states).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                rewards_tensor = torch.FloatTensor(rewards).to(device)
                dones_tensor = torch.FloatTensor(dones).to(device)

                q_values = policy_net(states_tensor)
                next_q_values = target_net(next_states_tensor)

                target_q_values = rewards_tensor + gamma * torch.max(next_q_values, dim=1)[0] * (1 - dones_tensor)

                optimizer.zero_grad()
                loss = nn.MSELoss()(q_values.gather(1, torch.LongTensor(actions).to(device).unsqueeze(1)), target_q_values.unsqueeze(1))
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward

        # epsilon 衰减
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 每隔一定的 episode 更新目标网络
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)

        # 计算滑动平均奖励（每 10 个 episode 取平均）
        if len(episode_rewards) >= 10:
            moving_avg_rewards.append(np.mean(episode_rewards[-10:]))
        else:
            moving_avg_rewards.append(np.mean(episode_rewards))

        # 打印当前 episode 的信息
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}")

    # 绘制奖励曲线
    plt.figure(figsize=(14, 10))
    plt.plot(np.array(episode_rewards), label='Episode Reward')
    plt.plot(np.array(moving_avg_rewards), label='Average Reward', linestyle='--', color='orange')
    plt.xlabel('Episode',fontsize=25)
    plt.ylabel('Total Reward',fontsize=25)
    plt.title('DQN Training - CartPole',fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

    return 




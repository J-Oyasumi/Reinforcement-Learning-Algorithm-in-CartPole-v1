import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# 定义Critic网络
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 定义Actor-Critic训练函数
def train_actor_critic(env, episodes, gamma=0.99, lr=1e-3):

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n

    # 初始化Actor和Critic网络
    actor = Actor(input_dim, output_dim)
    critic = Critic(input_dim)
    actor_optimizer = optim.Adam(actor.parameters(), lr=lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=lr)

    episode_rewards = []
    moving_avg_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 将状态转为Tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 通过Actor网络获取动作分布
            action_probs = actor(state_tensor)
            action = np.random.choice(output_dim, p=action_probs.detach().numpy()[0])

            # 执行动作并获得下一个状态和奖励
            next_state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            # 将下一状态转为Tensor
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # 计算Critic的值函数
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)

            # 计算TD目标和TD误差
            td_target = reward + gamma * next_value * (1 - done)
            td_error = td_target - value

            # 更新Critic网络
            critic_loss = td_error.pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # 更新Actor网络
            advantage = td_error.detach()  # 优势函数
            log_prob = torch.log(action_probs[0, action])
            actor_loss = -log_prob * advantage
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state

        episode_rewards.append(total_reward)
        if len(episode_rewards) >= 10:
            moving_avg_rewards.append(np.mean(episode_rewards[-10:]))
        else:
            moving_avg_rewards.append(np.mean(episode_rewards))


        print(f"Episode {episode + 1}, Total Reward: {total_reward}")


    plt.figure(figsize=(14, 10))
    plt.plot(np.array(episode_rewards), label='Episode Reward')
    plt.plot(np.array(moving_avg_rewards), label='Average Reward', linestyle='--', color='orange')
    plt.xlabel('Episode',fontsize=25)
    plt.ylabel('Total Reward',fontsize=25)
    plt.title('Actor-Critic Training - CartPole',fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

 
    return 

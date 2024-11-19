import numpy as np
import gym
import matplotlib.pyplot as plt

def train_qlearning(env, episodes, alpha=0.1, gamma=0.99, epsilon=1.0,epsilon_decay=0.95):
    # 离散化状态空间
    state_space_size = (50, 50, 20, 20)  
    action_space_size = env.action_space.n
    q_table = np.ones(state_space_size + (action_space_size,)) * 0.1 

    # 状态离散化函数
    def discretize_state(state):
        bins_arr = [
            np.linspace(-4.8, 4.8, 50),      
            np.linspace(-5, 5, 50),          
            np.linspace(-0.41887903, 0.41887903, 20),  
            np.linspace(-5, 5, 20),           
        ]
        return tuple(int(np.digitize(s, bins)) for s, bins in zip(state, bins_arr))

    # 用于记录每个 episode 的总奖励
    episode_rewards = []
    moving_avg_rewards = []

    # 训练循环
    for episode in range(episodes):
        state, _ = env.reset()  # 重置环境
        state = discretize_state(state)
        done = False
        total_reward = 0

        while not done:
            # epsilon-greedy 策略选择动作
            if np.random.rand() < epsilon:
                action = np.random.choice(action_space_size)  # 探索：随机选择动作
            else:
                action = np.argmax(q_table[state])  # 利用：选择 Q 值最大的动作

            # 执行动作并获得下一个状态和奖励
            next_state, reward, done,truncated,_= env.step(action)
            next_state = discretize_state(next_state)

            # 更新 Q 值
            q_table[state][action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            # 移动到下一个状态
            state = next_state
            total_reward += reward

        # 逐步降低 epsilon，减少探索，增加利用
        epsilon = max(0.01, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)  # 记录每个 episode 的总奖励
        print(f"Episode {episode+1}: Total Reward = {total_reward} epsilon={epsilon}")
    
        if len(episode_rewards) >= 10:
            moving_avg_rewards.append(np.mean(episode_rewards[-10:]))
        else:
            moving_avg_rewards.append(np.mean(episode_rewards))
        
    plt.figure(figsize=(14, 10))
    plt.plot(np.array(episode_rewards), label='Episode Reward')
    plt.plot(np.array(moving_avg_rewards), label='Average Reward', linestyle='--', color='orange')
    plt.xlabel('Episode',fontsize=25)
    plt.ylabel('Total Reward',fontsize=25)
    plt.title('QLearning Training - CartPole',fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

    return 


import numpy as np
import matplotlib.pyplot as plt

def random_(env):

    action_space_size = env.action_space.n

    episodes=100
    episode_rewards=[]

    for episode in range(episodes):
        state, _ = env.reset()  # 重置环境
        done = False
        total_reward = 0

        while not done:
            action=np.random.choice(action_space_size)
            next_state,reward,done,truncated,_=env.step(action)
            state=next_state
            total_reward+=reward

        episode_rewards.append(total_reward)  # 记录每个 episode 的总奖励
        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    plt.figure(figsize=(14, 10))
    plt.plot(np.array(episode_rewards), label='Episode Reward')
    plt.xlabel('Episode',fontsize=25)
    plt.ylabel('Total Reward',fontsize=25)
    plt.title('Random - CartPole',fontsize=25)
    plt.legend(fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()

    print(np.array(episode_rewards).mean())    

    # average reward 21.88
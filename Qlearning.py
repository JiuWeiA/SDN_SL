import gymnasium as gym
import numpy as np
import random
import time

# --- 基础配置 ---
seed_value = 421
random.seed(seed_value)
np.random.seed(seed_value)


# 1. 环境初始化
# 为了防止每一轮都自动弹出窗口，我们可以准备两个环境：
# 一个用于快速训练 (render_mode=None)，一个用于演示 (render_mode="human")
def create_env(render_mode=None):
    return gym.make(
        'FrozenLake-v1',
        map_name="8x8",
        is_slippery=False,
        render_mode=render_mode
    )


train_env = create_env(render_mode=None)
# 演示环境仅在需要时创建

# 超参数设置
learning_rate = 0.1
discount_factor = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
n_episodes = 1000

# 2. Q-table 初始化
num_states = train_env.observation_space.n
num_actions = train_env.action_space.n
Qtable = np.zeros((num_states, num_actions))


def choose_action(state, eps):
    """使用 epsilon-greedy 策略选择动作"""
    if random.uniform(0, 1) < eps:
        return train_env.action_space.sample()  # 探索
    else:
        return np.argmax(Qtable[state])  # 利用


print(f"开始训练 (静默模式)...")

# --- 训练循环 ---
for episode in range(n_episodes):
    state, _ = train_env.reset()
    done = False
    action = choose_action(state, epsilon)
    # 这一轮训练 (使用 train_env，无渲染)
    while not done:
        next_state, reward, terminated, truncated, _ = train_env.step(action)
        # 4. 奖励修正
        adj_reward = reward
        if terminated and reward == 0:
            adj_reward = -10.0
        elif reward == 1:
            adj_reward = 30.0
        else:
            adj_reward = -1.0

        # # 5. Q-table 更新
        # best_next_q = np.max(Qtable[next_state])
        # Qtable[state][action] = Qtable[state][action] + learning_rate * \
        #                         (adj_reward + discount_factor * best_next_q - Qtable[state][action])
        # state = next_state
        next_action = choose_action(next_state, epsilon)
        Qtable[state][action] = Qtable[state][action] + learning_rate * \
                                (adj_reward + discount_factor * Qtable[next_state][next_action] - Qtable[state][action])
        state = next_state
        action = next_action
        done = terminated or truncated

    # 衰减探索率
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # --- 定期演示阶段 ---
    # 每 200 轮展示一次
    if (episode + 1) % 200 == 0:
        print(f"第 {episode + 1} 轮训练完成，启动可视化演示...")
        # 临时创建一个带渲染窗口的环境进行演示
        demo_env = create_env(render_mode="human")
        demo_state, _ = demo_env.reset()
        demo_done = False
        step_count = 0

        while not demo_done and step_count < 20:
            demo_env.render()
            # 演示完全使用当前最优策略
            demo_action = np.argmax(Qtable[demo_state])
            demo_state, demo_reward, demo_term, demo_trun, _ = demo_env.step(demo_action)
            demo_done = demo_term or demo_trun
            step_count += 1

        demo_env.render()
        print(f"Episode {episode + 1}, Epsilon: {epsilon:.3f}, 结果: {'成功' if demo_reward == 1 else '失败'}")
        demo_env.close()  # 演示完关闭窗口，回到静默训练模式

print("\n所有训练结束。")
train_env.close()
import numpy as np
import random
import matplotlib.pyplot as plt

seed_value = 4211  # 你可以选择任何整数作为种子
# 设置 Python 的 random 模块种子
random.seed(seed_value)
np.random.seed(seed_value)

# 参数
alpha = 0.1
learning_rate = 0.05
discount_factor = 0.9
action_num = 3  # 下、左、右
state_num = 5  # 5x5 迷宫
start_state = 0
epsilon = 1  # 初始探索率
epsilon_min = 0.01  # 最小探索率
epsilon_decay = 0.995  # 衰减率
n_episodes = 2000  # 训练轮次

# Q-table 和奖励表初始化
Qtable = np.random.uniform(-1, 1, size=(state_num * state_num, action_num))  # 状态-动作 Q值表
Qtable_real = np.full((state_num * state_num, action_num), -1)

# 奖励和惩罚区域字典
reward_zones = {24: 30}  # 目标位置奖励
penalty_zones = {5: -10, 6: -10, 7: -10, 8: -10, 19: -10, 18: -10}  # 惩罚区域

# 将奖励和惩罚区域更新到奖励表
for pos, reward in reward_zones.items():
    Qtable_real[pos] = reward

for pos, penalty in penalty_zones.items():
    Qtable_real[pos] = penalty

# 无效动作列表
cannotaction_left = (0, 5, 10, 15, 20)
cannotaction_right = (4, 9, 14, 19, 24)
cannotaction_down = (20, 21, 22, 23, 24)


# 选择动作（ε-greedy策略）
def chose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice([0, 1, 2])  # 随机选择 3 个动作中的任意一个
    else:
        return np.argmax(Qtable[state])  # 选择 Q-table 中最大值对应的动作


# 状态转移函数
def step(action, state):
    if action == 0:  # 左
        if state not in cannotaction_left:
            next_state = state - 1
        else:
            next_state = state
    elif action == 1:  # 右
        if state not in cannotaction_right:
            next_state = state + 1
        else:
            next_state = state
    elif action == 2:  # 下
        if state not in cannotaction_down:
            next_state = state + 5
        else:
            next_state = state

    reward = Qtable_real[next_state][0]
    return next_state, reward


# 获取最优路径（从起始状态到目标状态）
def get_best_path():
    state = start_state
    path = [state]
    while state != 24:  # 直到达到目标状态
        action = np.argmax(Qtable[state])  # 选择最大 Q 值的动作
        next_state, _ = step(action, state)
        if next_state in path:
            return -1
        path.append(next_state)
        state = next_state
    return path


# 存储每一轮的最优路径
all_paths = []

# Q-learning 主训练循环
for episode in range(n_episodes):
    state = start_state
    done = False
    total_reward = 0

    while not done:
        action = chose_action(state)
        next_state, reward = step(action, state)

        # Q-table 更新
        Qtable[state][action] = (1 - learning_rate) * Qtable[state][action] + learning_rate * (
                    reward + discount_factor * np.max(Qtable[next_state]))

        state = next_state
        total_reward += reward

        # 如果到达终点
        if state == 24:
            done = True

    # 每 100 轮训练后保存最优路径
    if (episode + 1) % 200 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

        # 获取当前的最优路径并保存
        if get_best_path() != -1:
            best_path = get_best_path()
            all_paths.append(best_path)

    # 衰减 epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay  # 衰减 epsilon


# 在训练结束后一次性绘制所有最优路径
for episode, best_path in enumerate(all_paths):
    # 绘制迷宫和最优路径
    maze = np.zeros((state_num, state_num))  # 初始化迷宫
    for i in range(state_num):
        for j in range(state_num):
            pos = i * state_num + j
            if pos in reward_zones:  # 目标位置
                maze[i][j] = 2  # 目标位置用 2 标记
            elif pos in penalty_zones:  # 惩罚区域
                maze[i][j] = -1  # 惩罚区域用 -1 标记

    # 绘制迷宫的图像
    plt.figure(figsize=(6, 6))
    plt.imshow(maze, cmap='coolwarm', interpolation='nearest')

    # 绘制最优路径
    path_x = [p // state_num for p in best_path]
    path_y = [p % state_num for p in best_path]
    plt.plot(path_y, path_x, marker='o', color='r', linestyle='-', markersize=6, label="Best Path")

    plt.title(f"Optimal Path after {100 * (episode + 1)} episodes")
    plt.colorbar()
    plt.show()

# 最终的 Q-table
print("Final Q-table:")
print(Qtable)
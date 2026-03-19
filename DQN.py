import gymnasium as gym
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# =========================
# 基础配置
# =========================
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)
if torch.cuda.is_available():
    print("GPU名称:", torch.cuda.get_device_name(0))


# =========================
# DQN 网络定义
# =========================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# =========================
# 环境初始化
# =========================
def create_env(render_mode=None):
    return gym.make(
        'FrozenLake-v1',
        map_name="8x8",
        is_slippery=False,
        render_mode=render_mode
    )


train_env = create_env(render_mode=None)

# =========================
# 超参数
# =========================
learning_rate = 0.01
discount_factor = 0.99
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
batch_size = 256
memory_size = 10000
target_update_freq = 10
n_episodes = 1000

# 状态和动作维度
num_states = train_env.observation_space.n
num_actions = train_env.action_space.n

# =========================
# 初始化网络
# =========================
policy_net = DQN(num_states, num_actions).to(device)
target_net = DQN(num_states, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)


# =========================
# 状态处理：离散状态 -> One-hot
# =========================
def state_to_tensor(state):
    one_hot = np.zeros(num_states, dtype=np.float32)
    one_hot[state] = 1.0
    return torch.tensor(one_hot, dtype=torch.float32, device=device).unsqueeze(0)


# =========================
# 动作选择
# =========================
def choose_action(state_tensor, eps):
    if random.uniform(0, 1) < eps:
        return train_env.action_space.sample()
    else:
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            return q_values.argmax(dim=1).item()


# =========================
# 单步训练
# =========================
def train_step():
    if len(memory) < batch_size:
        return None

    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states).to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device).view(-1, 1)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.cat(next_states).to(device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)

    # 当前 Q 值
    current_q_values = policy_net(states).gather(1, actions).squeeze(1)

    # 目标 Q 值
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(dim=1)[0]
        expected_q_values = rewards + discount_factor * max_next_q_values * (1 - dones)

    # 损失
    loss = nn.MSELoss()(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


print("开始 DQN 训练 (静默模式)...")
print("policy_net 参数所在设备:", next(policy_net.parameters()).device)

# 用于记录训练信息
reward_history = []
loss_history = []

# =========================
# 训练循环
# =========================
for episode in range(n_episodes):
    state, _ = train_env.reset(seed=seed_value + episode)
    state_tensor = state_to_tensor(state)

    done = False
    episode_reward = 0
    losses = []
    steps = 0
    max_steps = 100  # 防止极端情况下单局太长

    while not done and steps < max_steps:
        action = choose_action(state_tensor, epsilon)
        next_state, reward, terminated, truncated, _ = train_env.step(action)

        # 奖励塑形
        if terminated and reward == 0:
            adj_reward = -10.0   # 掉洞
        elif reward == 1:
            adj_reward = 30.0    # 到终点
        else:
            adj_reward = -1.0    # 每走一步扣分

        next_state_tensor = state_to_tensor(next_state)
        is_done = terminated or truncated

        memory.append((state_tensor, action, adj_reward, next_state_tensor, is_done))

        state_tensor = next_state_tensor
        done = is_done
        episode_reward += adj_reward
        steps += 1

        loss_val = train_step()
        if loss_val is not None:
            losses.append(loss_val)

    reward_history.append(episode_reward)

    avg_loss = np.mean(losses) if losses else 0.0
    loss_history.append(avg_loss)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode + 1) % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if (episode + 1) % 100 == 0:
        recent_reward = np.mean(reward_history[-100:])
        print(
            f"Episode {episode + 1:4d} | "
            f"Reward={episode_reward:6.1f} | "
            f"AvgReward(100)={recent_reward:7.2f} | "
            f"Loss={avg_loss:.4f} | "
            f"Epsilon={epsilon:.3f}"
        )

    # 定期演示
    if (episode + 1) % 200 == 0:
        print(f"\n第 {episode + 1} 轮训练完成，演示当前 DQN 策略结果...")
        demo_env = create_env(render_mode="human")
        d_state, _ = demo_env.reset()
        d_done = False
        d_steps = 0

        while not d_done and d_steps < 30:
            d_tensor = state_to_tensor(d_state)

            with torch.no_grad():
                d_action = policy_net(d_tensor).argmax(dim=1).item()

            d_state, d_reward, d_term, d_trun, _ = demo_env.step(d_action)
            d_done = d_term or d_trun
            d_steps += 1

            time.sleep(0.15)

        demo_env.close()
        print("演示结束。\n")

print("\nDQN 训练结束。")
train_env.close()


# =========================
# 最终测试（不探索）
# =========================
print("开始最终测试（贪心策略）...")
test_env = create_env(render_mode="human")
state, _ = test_env.reset()
done = False
steps = 0
total_reward = 0

while not done and steps < 30:
    state_tensor = state_to_tensor(state)
    with torch.no_grad():
        action = policy_net(state_tensor).argmax(dim=1).item()

    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1
    time.sleep(0.2)

test_env.close()
print(f"最终测试结束，总原始奖励: {total_reward}, 步数: {steps}")
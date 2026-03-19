import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import time

# =========================
# 基础配置
# =========================
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前设备:", device)
if torch.cuda.is_available():
    print("GPU名称:", torch.cuda.get_device_name(0))


# =========================
# 1. 环境初始化
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
# 策略网络
# =========================
class PolicyNetwork(nn.Module):
    def __init__(self, n_states, n_actions):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 参数初始化
# =========================
n_states = train_env.observation_space.n
n_actions = train_env.action_space.n
learning_rate = 0.01
gamma = 0.99
max_steps_per_episode = 100
n_episodes = 8000


policy = PolicyNetwork(n_states, n_actions).to(device)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
print("模型所在设备:", next(policy.parameters()).device)


# =========================
# 状态编码
# =========================
def get_one_hot(state):
    """将整数状态转为 one-hot 张量，并放到 device 上"""
    oh = np.zeros(n_states, dtype=np.float32)
    oh[state] = 1.0
    return torch.tensor(oh, dtype=torch.float32, device=device)


print(f"开始训练 REINFORCE (8x8, 最长步数: {max_steps_per_episode})...")

for episode in range(n_episodes):
    state, _ = train_env.reset()

    ep_states = []
    ep_actions = []
    ep_rewards = []

    done = False
    steps = 0

    # =========================
    # 1. 交互阶段
    # =========================
    while not done and steps < max_steps_per_episode:
        s_tensor = get_one_hot(state)
        probs = policy(s_tensor)
        dist = Categorical(probs)

        action = dist.sample()

        next_state, reward, term, trun, _ = train_env.step(action.item())

        # 奖励修正
        if reward == 1:
            adj_reward = 30.0
        elif term:
            adj_reward = -10.0
        else:
            adj_reward = -0.05

        ep_states.append(s_tensor)
        ep_actions.append(action)
        ep_rewards.append(adj_reward)

        state = next_state
        done = term or trun
        steps += 1

    # =========================
    # 2. 更新阶段
    # =========================
    if len(ep_rewards) == 0:
        continue

    G = 0
    returns = []
    for r in reversed(ep_rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    # 回报归一化
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    optimizer.zero_grad()
    loss = torch.tensor(0.0, device=device)

    for s, a, g in zip(ep_states, ep_actions, returns):
        probs = policy(s)
        dist = Categorical(probs)
        loss = loss + (-dist.log_prob(a) * g)

    loss.backward()
    optimizer.step()

    # =========================
    # 3. 结果观察
    # =========================
    if (episode + 1) % 500 == 0:
        success = any(r > 1.0 for r in ep_rewards)
        print(
            f"回合 {episode + 1}: "
            f"步数={steps}, "
            f"损失={loss.item():.4f}, "
            f"任务{'成功' if success else '失败'}"
        )

    if (episode + 1) % 2000 == 0:
        print(f"\n第 {episode + 1} 轮，演示当前策略...")
        demo_env = create_env(render_mode="human")
        ds, _ = demo_env.reset()
        dd = False
        d_steps = 0

        while not dd and d_steps < max_steps_per_episode:
            time.sleep(0.05)

            with torch.no_grad():
                ds_tensor = get_one_hot(ds)
                da = policy(ds_tensor).argmax().item()

            ds, _, dt, dtr, _ = demo_env.step(da)
            dd = dt or dtr
            d_steps += 1

        demo_env.close()
        print("演示结束。\n")

train_env.close()

print("训练结束。")


# =========================
# 最终测试
# =========================
print("开始最终测试（贪心策略）...")
test_env = create_env(render_mode="human")
state, _ = test_env.reset()
done = False
steps = 0
total_reward = 0

while not done and steps < max_steps_per_episode:
    with torch.no_grad():
        s_tensor = get_one_hot(state)
        action = policy(s_tensor).argmax().item()

    state, reward, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated
    total_reward += reward
    steps += 1
    time.sleep(0.1)

test_env.close()
print(f"最终测试结束，总原始奖励: {total_reward}, 步数: {steps}")
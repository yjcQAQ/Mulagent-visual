"""
该游戏是一个典型的多智能体合作的类型，每个智能体（活塞）通过垂直上下移动将球移动到作边界，
每个智能体可以观察的范围为旁边两个智能体和上方的空间。

动作空间（一维）：
离散：{0:向下移动，1:保存不动，2:向上移动}
连续： value：[-1, 1]，1对应离散2的4倍，其余按[0-1]比例乘4计算

观察空间：
gym.spaces.Box(low=0, high=255, shape=(obs_height, self.piston_width * 3, 3)
三维矩阵，每个位置的RGB的强度

目标：
目标是让智能体（活塞）学会如何一起工作，以尽可能快地将球滚到左墙。
如果球向右移动，每个活塞代理都会得到负奖励，如果球向左移动则为正奖励，
并且在每个时间步都会收到少量的负奖励，以激励尽可能快地向左移动。

运行环境：
pip install ray==1.6.0
table_baselines3
pettingzoo
supersuit
matplotlib
"""

# 导入PPO算法，CnnPolicy自动调整神经网络的输入和输出层的大小以适应环境的观察和动作空间
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO

# 导入pistonball_v4环境
from pettingzoo.butterfly import pistonball_v4
import supersuit as ss

# 初始化环境
env = pistonball_v4.parallel_env(n_pistons=20, local_ratio=0, time_penalty=-0.1, continuous=True, random_drop=True,
                                 random_rotate=True, ball_mass=0.75, ball_friction=0.3, ball_elasticity=1.5,
                                 max_cycles=125)

# 环境的观察是全彩色图像。我们不需要颜色信息，
# 而且由于 3 个颜色通道，神经网络处理的计算成本比灰度图像高 3 倍。
# 我们可以通过用 SuperSuit 包装环境来解决这个问，这里使用B通道颜色
env = ss.color_reduction_v0(env, mode='B')

# 这个活塞的观察都是灰度的，但图像仍然非常大，
# 并且包含的信息比我们需要的要多。
# 让我们把它们缩小；84x84 是强化学习中常用的尺寸
env = ss.resize_v0(env, x_size=84, y_size=84)

# 由于球在运动，我们希望为策略网络提供一种简单的方法来查看它移动和加速的速度。
# 最简单的方法是将过去的几帧堆叠在一起作为每个观察的通道。
# 将 3 堆叠在一起提供了足够的信息来计算加速度，但 4 更标准
env = ss.frame_stack_v1(env, 3)

# 智能体环境中进行策略网络的参数共享
env = ss.pettingzoo_env_to_vec_env_v0(env)

# 设置环境以并行运行自身的多个版本，加快训练速度
env = ss.concat_vec_envs_v0(env, 8, num_cpus=4, base_class='stable_baselines3')
model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211,
            vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
# 模型训练
# model.learn(total_timesteps=3)

# 模型保存
# model.save("policy")

# Rendering
env = pistonball_v4.env()
env = ss.color_reduction_v0(env, mode='B')
env = ss.resize_v0(env, x_size=84, y_size=84)
env = ss.frame_stack_v1(env, 3)

# 恢复保存模型
model = PPO.load("policy")

# 在pycharm上智能体推演
# env.reset()
# for agent in env.agent_iter():
#     obs, reward, done, info = env.last()
#     act = model.predict(obs, deterministic=True)[0] if not done else None
#     env.step(act)
#     env.render()


# 在notebook上智能体推演
from IPython import display
import matplotlib.pyplot as plt
# %matplotlib inline
env.reset()
# 获得图像信息，可在notebook上显示
img = plt.imshow(env.render(mode='rgb_array'))
print(img)
print(plt.gcf())
# env.agent_iter 为所有智能体Done或者达到自己设定的最大iter
for agent in env.agent_iter():
    # 更新图像信息
    img.set_data(env.render(mode='rgb_array'))
    # 获得图片信息，可在notebook上显示
    display.display(plt.gcf())
    # display之前调用,clear_output以便在单元格中断时最终得到一个图而不是多个图
    display.clear_output(wait=True)

    # env.last：返回观察、奖励、完成和信息给当前的智能体
    # 奖励是该智能体上次的action
    obs, reward, done, info = env.last()
    act = model.predict(obs, deterministic=True)[0] if not done else None

    # 采取当前行动，Agent自动切换为下一个
    env.step(act)

    # 调用窗口可视化
    # env.render()

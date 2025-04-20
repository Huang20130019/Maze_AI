# -*- coding: utf-8 -*-
# 导入所有需要的库
import numpy as np
import random
import time
import collections
import pygame
import os # 用于文件检查和保存/加载
import pickle # 用于保存/加载缓冲区

# 导入 TensorFlow/Keras
try:
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
    from tensorflow.keras.optimizers import Adam
    import tensorflow as tf
    from keras import ops # Keras 3 ops 用于兼容 Functional API
except ImportError:
    print("Error: TensorFlow/Keras is not installed. Please run 'pip install tensorflow'.")
    exit()


# 定义网格单元格的常量
CELL_PATH = 0      # 通路
CELL_WALL = 1      # 墙壁
CELL_START = 2     # 起点
CELL_TREASURE = 3  # 宝藏
# 注意：CELL_AGENT 在 get_state 返回的状态地图中不再使用，因为我们回到了局部视图

# 定义行动的常量 (上、下、左、右)
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
NUM_ACTIONS = 4    # 总行动数量

# 定义局部视图的尺寸 (例如 10x10)
VIEW_SIZE = 10

# --- 定义文件保存路径 (全局作用域) ---
DQN_WEIGHTS_FILENAME = "dqn_maze_weights.h5"
DQN_BUFFER_FILENAME = "dqn_maze_replay_buffer.pkl"
DQN_TRAINING_RECORD_FILENAME = "dqn_maze_training_rewards.csv" # 定义奖励记录文件名
# --- 结束定义 ---


# --- START: MazeEnv Class (迷宫环境) ---
# 这个类包含迷宫的所有规则和状态管理
class MazeEnv:
    def __init__(self, maze_layout):
        """
        初始化迷宫环境
        :param maze_layout: 一个二维列表或 numpy 数组，表示迷宫的布局
                          例如: [[2, 0, 1], [0, 1, 0], [1, 0, 3]]
        """
        self.maze = np.array(maze_layout, dtype=int)
        # 注意：这里的 height 和 width 是生成后的迷宫网格的实际尺寸 (2*概念高度+1, 2*概念宽度+1)
        self.height, self.width = self.maze.shape

        # 找到起点和宝藏的所有位置
        start_positions = list(zip(*np.where(self.maze == CELL_START)))
        treasure_positions = list(zip(*np.where(self.maze == CELL_TREASURE)))

        if not start_positions:
            raise ValueError("Maze must have a start point (value 2)")
        if not treasure_positions:
             raise ValueError("Maze must have at least one treasure point (value 3)")

        # 假设只有一个起点，并找到它的 (行, 列)
        self.start_pos = start_positions[0]
        # 假设只有一个宝藏，并找到它的 (行, 列)
        self.treasure_pos = treasure_positions[0]

        self.agent_pos = None # 代理当前位置，reset时设置
        self.game_over = False # 游戏结束标志 (布尔属性)
        self.reward = 0 # 上一步获得的奖励 (仅记录最后一步的奖励)
        self.total_reward = 0 # 累积奖励
        self.steps_taken = 0 # 已采取的步数

    def reset(self):
        """
        重置环境，将代理放回起点
        :return: 初始状态 (代理位置元组) - 注意 AI 将调用 get_state() 获取网格状态
        """
        self.agent_pos = self.start_pos
        self.game_over = False
        self.reward = 0
        self.total_reward = 0
        self.steps_taken = 0
        # Reset 返回代理的 (行, 列) 位置元组
        return self.agent_pos

    def step(self, action):
        """
        根据行动更新代理位置和游戏状态
        :param action: 行动常量 (ACTION_UP, ACTION_DOWN, etc.)
        :return: next_state_tuple (下一个代理位置元组), reward (获得的奖励), done (游戏是否结束), info (额外信息字典)
        """
        if self.game_over:
            return self.agent_pos, 0, True, {"message": "Game is already over"}

        r, c = self.agent_pos

        reward = -1

        next_r, next_c = r, c

        if action == ACTION_UP:
            next_r -= 1
        elif action == ACTION_DOWN:
            next_r += 1
        elif action == ACTION_LEFT:
            next_c -= 1
        elif action == ACTION_RIGHT:
            next_c += 1
        else:
            self.steps_taken += 1
            self.total_reward += -5
            return self.agent_pos, -5, self.game_over, {"message": "Invalid action value"}

        if (0 <= next_r < self.height and
            0 <= next_c < self.width and
            self.maze[next_r, next_c] != CELL_WALL):
            self.agent_pos = (next_r, next_c)
        else:
            reward = -5

        self.steps_taken += 1
        self.total_reward += reward

        if self.agent_pos == self.treasure_pos:
            self.game_over = True
            treasure_reward = 100
            reward += treasure_reward
            self.total_reward += treasure_reward

        done = self.game_over

        # step 返回下一个代理位置的 (行, 列) 元组
        return self.agent_pos, reward, done, {}

    def render(self):
        """
        在控制台打印迷宫的当前状态 (代理位置) - 用于文本界面
        """
        display_maze = np.copy(self.maze).astype(str)

        display_maze[display_maze == str(CELL_PATH)] = '.'
        display_maze[display_maze == str(CELL_WALL)] = '#'
        display_maze[display_maze == str(CELL_START)] = 'S'
        display_maze[display_maze == str(CELL_TREASURE)] = 'T'

        r, c = self.agent_pos
        if (r, c) == self.treasure_pos and not self.game_over:
             display_maze[r, c] = 'A'
        elif (r, c) == self.treasure_pos and self.game_over:
             display_maze[r,c] = 'T' # 游戏结束在宝藏，显示T
        else:
             display_maze[r, c] = 'A' # 代理位置


        print("--- Maze State (Text) ---")
        for row in display_maze:
            print(" ".join(row))
        print(f"Agent Pos: {self.agent_pos}, Steps: {self.steps_taken}, Total Reward: {self.total_reward}")
        print("------------------")


    # --- START: MODIFIED get_state method (返回局部网格视图) ---
    # 确保 VIEW_SIZE 常量已定义
    def get_state(self):
        """
        获取包含代理周围 VIEW_SIZE x VIEW_SIZE 局部区域的网格信息作为状态
        这个状态将是一个 VIEW_SIZE x VIEW_SIZE 的 numpy 数组
        用 CELL_WALL 进行边界填充
        :return: 代理周围 VIEW_SIZE x VIEW_SIZE 的局部网格视图 (numpy 数组)
        """
        r, c = self.agent_pos
        view_half = VIEW_SIZE // 2 # 视图窗口半径

        # 创建一个 VIEW_SIZE x VIEW_SIZE 的网格，用墙壁进行边界填充
        local_view = np.full((VIEW_SIZE, VIEW_SIZE), CELL_WALL, dtype=int)

        # 遍历局部视图的每个单元格
        for i in range(VIEW_SIZE):
            for j in range(VIEW_SIZE):
                # 计算局部视图单元格对应的全局迷宫坐标
                maze_r = r - view_half + i
                maze_c = c - view_half + j

                # 检查全局迷宫坐标是否在迷宫范围内
                if 0 <= maze_r < self.height and 0 <= maze_c < self.width:
                    # 如果在范围内，将迷宫中的实际单元格值赋给局部视图
                    local_view[i, j] = self.maze[maze_r, maze_c]
                # else:
                    # 如果超出范围，保持为墙壁 (局部视图默认初始化为墙壁)

        # 在这个状态表示中，代理自己的位置信息隐含在它是视图的中心
        # 视图中的 CELL_START, CELL_TREASURE 仍然表示它们在局部视图内的位置

        return local_view # 返回局部网格视图 (VIEW_SIZE x VIEW_SIZE numpy 数组)
    # --- END: MODIFIED get_state method ---


    def is_goal_state(self, state_tuple):
        """
        检查给定的【代理位置元组】是否是目标状态 (宝藏位置)
        注意：这个方法接收的是【代理的位置元组】，不是局部视图
        :param state_tuple: 要检查的状态 (代理的 (行, 列) 元组)
        :return: True 如果是宝藏位置，否则 False
        """
        return state_tuple == self.treasure_pos

    def get_possible_actions(self, state_tuple):
        """
        获取在给定【代理位置元组】下所有可能的行动列表 (上、下、左、右)
        注意：这个方法接收的是【代理的位置元组】，不是局部视图
        """
        return [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
# --- END: MazeEnv Class ---


# --- START: ReplayBuffer Class ---
# 这个类用于存储代理与环境互动的经验，供 DQN 学习时回放

class ReplayBuffer:
    def __init__(self, capacity):
        """
        初始化经验回放缓冲区
        :param capacity: 缓冲区的最大容量
        """
        self.buffer = collections.deque(maxlen=capacity) # 使用双端队列实现固定容量缓冲区

    def push(self, state, action, reward, next_state, done):
        """
        将一次经验添加到缓冲区
        :param state: 当前状态 (numpy 数组 - 局部视图)
        :param action: 采取的行动 (整数)
        :param reward: 获得的奖励 (浮点数)
        :param next_state: 下一个状态 (numpy 数组 - 局部视图)
        :param done: 游戏是否结束 (布尔值)
        """
        # 经验元组 (state, action, reward, next_state, done)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一批经验
        :param batch_size: 采样的批量大小
        :return: 批量经验的元组 (states, actions, rewards, next_states, dones)
                 每个元素都是一个 numpy 数组
        """
        if len(self.buffer) < batch_size:
            return None # 返回 None 表示无法采样

        # 随机选择 batch_size 个经验的索引
        batch_indices = random.sample(range(len(self.buffer)), batch_size)

        # 从缓冲区中提取对应的经验
        batch = [self.buffer[i] for i in batch_indices]

        # 将批量经验按类型分组
        states, actions, rewards, next_states, dones = zip(*batch)

        # 将采样结果转换为 numpy 数组
        # 状态 (局部视图) 已经是 numpy 数组，zip 后是列表，需要 np.array 堆叠起来
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32) # 奖励使用浮点类型
        dones = np.array(dones, dtype=np.float32) # Done 标志使用浮点类型 (0.0 或 1.0)，方便计算 Target Q

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """返回缓冲区当前的经验数量"""
        return len(self.buffer)

    # 添加保存和加载缓冲区的功能
    def save_buffer(self, filename): # filename 作为参数传入
        """将缓冲区内容保存到文件"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(list(self.buffer), f) # 将 deque 转换为 list 后保存
            #print(f"Replay buffer saved to {filename}") # 训练时频繁保存可能打印过多
        except Exception as e:
            print(f"Error saving replay buffer: {e}")


    def load_buffer(self, filename): # filename 作为参数传入
        """从文件加载缓冲区内容"""
        if not os.path.exists(filename):
            print(f"Replay buffer file not found: {filename}")
            return False
        try:
            with open(filename, 'rb') as f:
                loaded_list = pickle.load(f)
                # 检查加载的列表长度是否超过缓冲区容量
                if len(loaded_list) > self.buffer.maxlen:
                     loaded_list = loaded_list[-self.buffer.maxlen:] # 截断以匹配容量

                self.buffer = collections.deque(loaded_list, maxlen=self.buffer.maxlen) # 加载后转换回 deque
            print(f"Replay buffer loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading replay buffer: {e}")
            # 如果加载失败，可能文件损坏，缓冲区保持当前状态 (通常是空的或部分加载的)
            # try: os.remove(filename) except: pass
            return False
# --- END: ReplayBuffer Class ---


# --- START: DQNAgent Class ---
# 这个类实现了 DQN 算法

class DQNAgent:
    def __init__(self, state_shape, num_actions,
                 learning_rate=0.001,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_steps=50000, # Epsilon 按总步数衰减
                 replay_buffer_capacity=50000, # 缓冲区容量
                 batch_size=32,
                 target_update_frequency=500, # 目标网络更新频率 (总步数)
                 random_seed=None):
        """
        初始化 DQN 代理
        :param state_shape: 状态的形状 (例如 (VIEW_SIZE, VIEW_SIZE, 1))
        :param num_actions: 行动的数量
        :param learning_rate: 神经网络学习率
        :param discount_factor: 折扣因子 (gamma)
        :param epsilon_start: 探索率起始值
        :param epsilon_end: 探索率最终值
        :param epsilon_decay_steps: epsilon 从 start 衰减到 end 需要多少个【总步数】
        :param replay_buffer_capacity: 经验回放缓冲区的容量
        :param batch_size: 训练时从缓冲区采样的批量大小
        :param target_update_frequency: 目标网络参数更新频率 (每隔多少个【总步数】更新一次)
        :param random_seed: 随机种子 (用于重现)
        """
        # 确保 TensorFlow/Keras 已导入

        if random_seed is not None:
             tf.random.set_seed(random_seed)
             np.random.seed(random_seed)
             random.seed(random_seed)


        self.state_shape = state_shape # (VIEW_SIZE, VIEW_SIZE, 1)
        self.num_actions = num_actions # 4

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start # 当前探索率
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0 # 记录总步数，用于 epsilon 衰减和目标网络更新

        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        # 创建 Primary Network 和 Target Network
        self.primary_network = self.build_model()
        self.target_network = self.build_model()
        # 初始时，目标网络的权重与主网络相同
        self.target_network.set_weights(self.primary_network.get_weights())

        # 创建经验回放缓冲区
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)


    def build_model(self):
        """
        构建神经网络模型 (Primary Network 和 Target Network 使用相同结构)
        输入层形状是 self.state_shape (例如 (VIEW_SIZE, VIEW_SIZE, 1))
        输出层形状是 (NUM_ACTIONS,)
        """
        # 使用 Functional API 创建模型
        input_layer = Input(shape=self.state_shape)

        # 确保输入的每个元素都是浮点类型
        # 使用 keras.ops.cast 兼容 Functional API
        x = ops.cast(input_layer, 'float32')

        # 卷积层用于处理网格状的输入 (局部视图)
        # input_shape 在第一层 Input 中指定
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Flatten()(x) # 将卷积层的输出展平为一维向量

        # 全连接层
        x = Dense(128, activation='relu')(x)
        # 输出层：每个行动一个输出单元，表示对应行动的 Q 值
        output_layer = Dense(self.num_actions, activation='linear')(x) # DQN 输出 Q 值，激活函数通常是 linear

        model = Model(inputs=input_layer, outputs=output_layer)

        # 使用 Adam 优化器
        optimizer = Adam(learning_rate=self.learning_rate)

        # 编译模型
        # loss='mse' (均方误差) 是 DQN 常见的损失函数
        model.compile(optimizer=optimizer, loss='mse')

        # print("DQN Model Summary:")
        # model.summary() # 打印模型结构摘要

        return model

    def get_action(self, state, evaluation=False):
        """
        根据当前状态和 epsilon-greedy 策略选择一个行动
        :param state: 代理的当前状态 (VIEW_SIZE x VIEW_SIZE numpy 数组)
        :param evaluation: 是否处于评估模式 (不进行探索，epsilon=0)
        :return: 选择的行动常量 (ACTION_UP, etc.)
        """
        # 在评估模式下，不进行探索
        if evaluation:
            epsilon = 0.0
        else:
            # Epsilon 按总步数进行线性衰减
            epsilon = max(self.epsilon_end, self.epsilon_start - (self.epsilon_start - self.epsilon_end) * self.total_steps / self.epsilon_decay_steps)
            self.epsilon = epsilon # 更新代理当前的 epsilon 值

        # epsilon-greedy 策略
        if random.random() < epsilon:
            # 探索：随机选择一个行动
            action = random.randint(0, self.num_actions - 1)
        else:
            # 利用：选择当前状态下预测 Q 值最高的行动
            # 神经网络期望批量输入，所以需要将状态 reshape 成 (1, VIEW_SIZE, VIEW_SIZE)
            # 并确保输入形状匹配模型的 input_shape (height, width, channels)
            # state 是 (VIEW_SIZE, VIEW_SIZE)
            state_input = np.expand_dims(state, axis=0) # 添加批量维度 -> (1, VIEW_SIZE, VIEW_SIZE)
            state_input = np.expand_dims(state_input, axis=-1) # 添加通道维度 -> (1, VIEW_SIZE, VIEW_SIZE, 1)

            # 使用 Primary Network 预测 Q 值
            q_values = self.primary_network.predict(state_input, verbose=0) # verbose=0 suppresses prediction logs

            # 找到预测 Q 值最高的行动的索引
            action = np.argmax(q_values[0]) # q_values 是 (1, NUM_ACTIONS) 形状，取第一个元素的 argmax

        return action

    def learn(self):
        """
        从经验回放缓冲区中采样批量经验，并更新 Primary Network 的权重
        只有当缓冲区中的经验足够多时才进行学习
        """
        # 如果缓冲区中的经验不足一个批量大小，不进行学习
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从缓冲区中采样一个批量经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 准备输入数据和目标 Q 值
        # 神经网络期望输入形状是 (batch_size, height, width, channels)
        # states 和 next_states 已经是 (batch_size, VIEW_SIZE, VIEW_SIZE) numpy 数组
        states_input = np.expand_dims(states, axis=-1) # 添加通道维度 -> (batch_size, VIEW_SIZE, VIEW_SIZE, 1)
        next_states_input = np.expand_dims(next_states, axis=-1) # 添加通道维度 -> (batch_size, VIEW_SIZE, VIEW_SIZE, 1)


        # 使用 Primary Network 预测当前状态的 Q 值 (用于计算损失)
        # Q(s_t, a_t)
        predicted_q_values = self.primary_network.predict(states_input, verbose=0)

        # 使用 Target Network 预测下一个状态的最大 Q 值 (用于计算目标 Q)
        # max_a Q(s_{t+1}, a)
        next_q_values = self.target_network.predict(next_states_input, verbose=0)
        max_next_q = np.max(next_q_values, axis=1) # 找到每个批量样本的下一个状态的最大 Q 值


        # 计算目标 Q 值
        # Target Q = reward + discount_factor * max_a Q(s_{t+1}, a) (如果 not done)
        # Target Q = reward (如果 done)
        # dones 是浮点类型 (0.0 或 1.0)，直接用于乘法
        target_q_values = rewards + self.discount_factor * max_next_q * (1 - dones)

        # 创建用于训练的目标 Q 值数组
        # 目标 Q 值形状与 predicted_q_values 相同
        # 除了采取的行动的 Q 值被更新为 Target Q，其他行动的 Q 值保持不变
        target_qs_for_training = predicted_q_values.copy() # 复制预测的 Q 值

        # 根据批量样本的行动，更新目标 Q 值中对应行动的值
        # np.arange(self.batch_size) 创建一个从 0 到 batch_size-1 的数组，代表批量中的样本索引
        target_qs_for_training[np.arange(self.batch_size), actions] = target_q_values

        # 训练 Primary Network
        # states_input 是神经网络的输入
        # target_qs_for_training 是神经网络的目标输出 (用于计算损失)
        self.primary_network.fit(states_input, target_qs_for_training, batch_size=self.batch_size, verbose=0) # verbose=0 suppresses training logs


        # 更新总步数 (每次学习使用一个批量)
        self.total_steps += self.batch_size


        # 周期性更新目标网络
        if self.total_steps % self.target_update_frequency == 0:
            self.update_target_network()


    # <--- 注意这里，update_target_network 方法的缩进应该在这里，与 learn 方法同级
    def update_target_network(self):
        """将 Primary Network 的权重复制到 Target Network"""
        self.target_network.set_weights(self.primary_network.get_weights())
        print(f"--- Updating target network at total step {self.total_steps} ---")


    def save_model(self, weights_filename, buffer_filename):
        """保存 Primary Network 的权重和经验回放缓冲区"""
        try:
            self.primary_network.save_weights(weights_filename)
            # print(f"Model weights saved to {weights_filename}") # 训练时频繁保存可能打印过多
        except Exception as e:
             print(f"Error saving model weights: {e}")

        self.replay_buffer.save_buffer(buffer_filename) # 调用 ReplayBuffer 的保存方法


    def load_model(self, weights_filename, buffer_filename):
        """加载 Primary Network 的权重和经验回放缓冲区"""
        weights_loaded = False
        buffer_loaded = False

        # 加载模型权重
        if os.path.exists(weights_filename):
            try:
                # 在加载权重之前，需要先构建模型
                # 确保模型结构与保存的权重匹配
                # self.primary_network = self.build_model() # 模型已经在 __init__ 中构建
                self.primary_network.load_weights(weights_filename)
                self.target_network.set_weights(self.primary_network.get_weights()) # 目标网络也更新
                print(f"Model weights loaded from {weights_filename}")
                weights_loaded = True
            except Exception as e:
                print(f"Error loading model weights from {weights_filename}: {e}")
                # 如果加载失败，模型权重保持初始状态

        # 加载经验回放缓冲区
        buffer_loaded = self.replay_buffer.load_buffer(buffer_filename) # 调用 ReplayBuffer 的加载方法

        # TODO: 如果需要，可以在这里加载 total_steps 和 epsilon，以精确恢复训练进度
        # 这需要将 total_steps 和 epsilon 也保存和加载


        return weights_loaded or buffer_loaded # 只要有一个加载成功就返回 True

# --- END: DQNAgent Class ---


# --- START: Random Maze Generation Function ---
def generate_maze(height, width, random_seed=None):
    """
    使用递归回溯法生成一个随机迷宫布局
    :param height: 概念上的迷宫高度 (单元格数量)
    :param width: 概念上的迷高 (单元格数量)
    :param random_seed: 随机种子，用于生成固定的迷宫 (可选)
    :return: 生成的迷宫布局 (numpy 数组，尺寸为 (2*height+1, 2*width+1))
    """
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    grid_height = 2 * height + 1
    grid_width = 2 * width + 1

    maze = np.full((grid_height, grid_width), CELL_WALL, dtype=int)

    stack = []
    visited = np.zeros((height, width), dtype=bool)

    start_r_concept, start_c_concept = random.randint(0, height - 1), random.randint(0, width - 1)
    stack.append((start_r_concept, start_c_concept))
    visited[start_r_concept, start_c_concept] = True
    maze[start_r_concept * 2 + 1, start_c_concept * 2 + 1] = CELL_PATH

    directions = [
        ((-1, 0), (-1, 0)),  # Up (concept_dr, concept_dc), (grid_dr, grid_dc for wall)
        ((1, 0), (1, 0)),   # Down
        ((0, -1), (0, -1)),  # Left
        ((0, 1), (0, 1))    # Right
    ]

    while stack:
        curr_r_concept, curr_c_concept = stack[-1]

        neighbors = []
        shuffled_directions = list(directions)
        random.shuffle(shuffled_directions)

        for (dr_concept, dc_concept), (dr_grid, dc_grid) in shuffled_directions:
            next_r_concept, next_c_concept = curr_r_concept + dr_concept, curr_c_concept + dc_concept
            wall_r_grid, wall_c_grid = curr_r_concept * 2 + 1 + dr_grid, curr_c_concept * 2 + 1 + dc_grid

            if (0 <= next_r_concept < height and
                0 <= next_c_concept < width and
                not visited[next_r_concept, next_c_concept]):
                 next_r_grid, next_c_grid = next_r_concept * 2 + 1, next_c_concept * 2 + 1
                 neighbors.append(((next_r_concept, next_c_concept), (next_r_grid, next_c_grid), (wall_r_grid, wall_c_grid)))

        if neighbors:
            (next_r_concept, next_c_concept), (next_r_grid, next_c_grid), (wall_r_grid, wall_c_grid) = random.choice(neighbors)

            visited[next_r_concept, next_c_concept] = True
            maze[next_r_grid, next_c_grid] = CELL_PATH
            maze[wall_r_grid, wall_c_grid] = CELL_PATH

            stack.append((next_r_concept, next_c_concept))

        else:
            stack.pop()

    # --- 放置起点和宝藏 ---
    path_cells_grid = [(r, c) for r in range(grid_height) for c in range(grid_width) if maze[r, c] == CELL_PATH]

    if len(path_cells_grid) < 2:
        print("Warning: Not enough path cells to place Start and Treasure separately. Placing them at the same spot.")
        start_pos_grid = path_cells_grid[0] if path_cells_grid else (1,1)
        treasure_pos_grid = start_pos_grid
    else:
        start_pos_grid, treasure_pos_grid = random.sample(path_cells_grid, 2)

    maze[start_pos_grid] = CELL_START
    maze[treasure_pos_grid] = CELL_TREASURE

    print(f"Generated maze (concept: {height}x{width}, grid: {grid_height}x{grid_width})")
    print(f"Start at {start_pos_grid}, Treasure at {treasure_pos_grid}")

    return maze

# --- END: Random Maze Generation Function ---


# --- START: Training Record Save Function ---
def save_training_record(rewards, filename):
    """
    将训练过程中的每个回合奖励保存到文本文件 (CSV 格式)
    """
    try:
        with open(filename, 'w', newline='') as f:
            f.write("Episode,TotalReward\n")
            for i, reward in enumerate(rewards):
                f.write(f"{i+1},{reward}\n")
        print(f"Training rewards saved to {filename}")
    except Exception as e:
        print(f"Error saving training rewards to {filename}: {e}")
# --- END: Training Record Save Function ---


# --- START: Pygame Visualization Code ---

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (150, 150, 150)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (255, 0, 0)
COLOR_YELLOW = (255, 255, 0)
COLOR_BLUE = (0, 0, 255)


CELL_PIXELS = 30
MARGIN_PIXELS = 3


def draw_maze_pygame(screen, maze_env, WINDOW_WIDTH_PYGAME, WINDOW_HEIGHT_PYGAME):
    """
    使用 Pygame 绘制迷宫的当前状态
    注意：这个函数根据 env.maze 和 env.agent_pos 进行绘制，而不是根据 env.get_state() 返回的状态地图
    """
    screen.fill(COLOR_BLACK)

    for r in range(maze_env.height):
        for c in range(maze_env.width):
            cell_value = maze_env.maze[r, c] # 使用原始迷宫布局

            color = COLOR_GREEN
            if cell_value == CELL_WALL:
                color = COLOR_GRAY
            elif cell_value == CELL_START:
                color = COLOR_RED
            elif cell_value == CELL_TREASURE:
                color = COLOR_YELLOW

            x = MARGIN_PIXELS + c * (CELL_PIXELS + MARGIN_PIXELS)
            y = MARGIN_PIXELS + r * (CELL_PIXELS + MARGIN_PIXELS)
            rect = pygame.Rect(x, y, CELL_PIXELS, CELL_PIXELS)

            pygame.draw.rect(screen, color, rect)

    # 绘制代理 (根据代理的当前位置绘制蓝色圆圈)
    agent_r, agent_c = maze_env.agent_pos
    center_x = MARGIN_PIXELS + agent_c * (CELL_PIXELS + MARGIN_PIXELS) + CELL_PIXELS // 2
    center_y = MARGIN_PIXELS + agent_r * (CELL_PIXELS + MARGIN_PIXELS) + CELL_PIXELS // 2
    agent_radius = CELL_PIXELS // 3
    pygame.draw.circle(screen, COLOR_BLUE, (center_x, center_y), agent_radius)


    if maze_env.game_over:
        if not pygame.font.get_init():
             pygame.font.init()
        font_size = min(74, WINDOW_WIDTH_PYGAME // 10)
        font = pygame.font.Font(None, font_size)
        text = font.render("Treasure Found!", True, COLOR_WHITE)
        text_rect = text.get_rect(center=(WINDOW_WIDTH_PYGAME // 2, WINDOW_HEIGHT_PYGAME // 2))
        screen.blit(text, text_rect)


def run_maze_visualization(maze_layout, ai_agent=None):
    """
    运行迷宫环境的 Pygame 可视化
    :param maze_layout: 迷宫布局 (numpy 数组)
    :param ai_agent: 可选的 AI 代理对象 (需要有 get_action(state, evaluation=True) 方法)
                     如果为 None，将使用随机行动
    """
    print("Starting Maze Pygame visualization...")

    pygame.init()

    env = MazeEnv(maze_layout)

    WINDOW_WIDTH_PYGAME = env.width * CELL_PIXELS + (env.width + 1) * MARGIN_PIXELS
    WINDOW_HEIGHT_PYGAME = env.height * CELL_PIXELS + (env.height + 1) * MARGIN_PIXELS

    min_window_size = 200
    WINDOW_WIDTH_PYGAME = max(WINDOW_WIDTH_PYGAME, min_window_size)
    WINDOW_HEIGHT_PYGAME = max(WINDOW_HEIGHT_PYGAME, min_window_size)

    screen = pygame.display.set_mode((WINDOW_WIDTH_PYGAME, WINDOW_HEIGHT_PYGAME))
    pygame.display.set_caption("Maze Treasure Hunt Visualization")

    clock = pygame.time.Clock()

    state_tuple = env.reset() # Reset returns (row, col) tuple
    # current_agent_state = env.get_state() # Get initial grid state if needed by agent

    running = True
    game_speed_delay = 0.05 # 控制可视化速度
    last_action_time = time.time()

    print("Visualization ready. Close window to exit.")
    if ai_agent is None:
        print("Using random actions.")
    else:
         print(f"Using provided AI agent ({type(ai_agent).__name__}).")


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not env.game_over:
             current_time = time.time()
             # 控制行动速度
             if current_time - last_action_time > game_speed_delay:
                 action = None
                 if ai_agent is not None:
                      # 获取当前环境状态 (局部视图 - 新的状态表示)
                      current_agent_state = env.get_state()
                      # 将局部视图状态传递给 AI 代理获取行动 (评估模式)
                      action = ai_agent.get_action(current_agent_state, evaluation=True) # 在可视化时使用评估模式
                 else:
                      # 没有 AI 代理，使用随机行动
                      action = random.randint(0, NUM_ACTIONS - 1)


                 if action is not None:
                      # 执行行动，env.step 返回 (行, 列) 元组作为下一个状态
                      next_state_tuple, reward, done, info = env.step(action)
                      # 注意：这里的 next_state_tuple 是旧的状态表示 (行, 列)

                      last_action_time = current_time

                      if done:
                           print("Game finished! Total Reward:", env.total_reward)
                           # time.sleep(1)
                           # state_tuple = env.reset() # Reset env
                           # current_agent_state = env.get_state() # Get initial grid state for next round


        draw_maze_pygame(screen, env, WINDOW_WIDTH_PYGAME, WINDOW_HEIGHT_PYGAME)
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    print("Pygame visualization finished.")
# --- END: Pygame Visualization Code ---


# --- START: DQN Training Function (AI训练) ---
# 这个函数执行 DQN 强化学习的训练过程

def train_dqn_agent(maze_layout, num_episodes=5000, max_steps_per_episode=200):
    """
    运行 DQN 代理的训练过程
    :param maze_layout: 迷宫布局 (通常由 generate_maze 函数生成)
    :param num_episodes: 训练回合数
    :param max_steps_per_episode: 每个回合的最大步数
    :return: 每个回合的总奖励列表
    """
    print("Starting DQN agent training...")

    env = MazeEnv(maze_layout)

    # 定义 DQN 代理的超参数
    agent_params = {
        "state_shape": (VIEW_SIZE, VIEW_SIZE, 1), # 状态形状 (局部视图尺寸 x 1 通道)
        "num_actions": NUM_ACTIONS,
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay_steps": num_episodes * max_steps_per_episode * 0.8, # Epsilon 在总步数的前 80% 衰减
        "replay_buffer_capacity": 50000,
        "batch_size": 32,
        "target_update_frequency": 500, # 目标网络更新频率 (总步数)
        "random_seed": None # 设置训练种子
    }

    # 创建 DQN 代理实例
    agent = DQNAgent(**agent_params)

    # 尝试加载模型权重和缓冲区
    model_loaded = agent.load_model(DQN_WEIGHTS_FILENAME, DQN_BUFFER_FILENAME)
    if model_loaded:
        print("Resuming training from saved files.")
    else:
        print("No saved model or buffer found. Starting training from scratch.")

    episode_rewards = [] # 记录每个回合的总奖励
    # TODO: 如果加载了训练记录文件，可以将之前的奖励添加到这里，以便绘制完整的学习曲线
    # 目前 save_training_record 是覆盖写入，如果需要追加，需要修改 load/save

    # 训练主循环 (按回合进行)
    for episode in range(num_episodes):
        # env.reset() 返回 (行, 列) 元组
        state_tuple = env.reset()
        # 获取回合开始时的初始状态 (局部视图)，这是代理需要的状态
        current_agent_state = env.get_state()

        episode_reward = 0
        done = False
        steps_this_episode = 0

        # 回合内的步骤循环
        while not done and steps_this_episode < max_steps_per_episode:
            # 获取代理的行动 (训练模式)
            # total_steps 由 agent.learn() 内部更新并用于 epsilon 衰减
            action = agent.get_action(current_agent_state, evaluation=False)

            # 在环境中执行行动
            # env.step 返回下一个代理位置的 (行, 列) 元组 和 奖励等信息
            next_state_tuple, reward, done, _ = env.step(action)

            # 获取下一个状态的【局部视图】
            next_agent_state = env.get_state()

            # 将经验添加到经验回放缓冲区
            agent.replay_buffer.push(current_agent_state, action, reward, next_agent_state, done)

            # 如果缓冲区经验足够，进行一次学习
            if len(agent.replay_buffer) >= agent.batch_size: # 可以设置一个最小学习经验数量，例如 buffer_capacity * 0.1
                 agent.learn() # learn 方法内部会采样批量并更新网络，也会更新 total_steps 和 epsilon，并周期性更新目标网络

            # 更新当前状态变量为下一个状态的局部视图
            current_agent_state = next_agent_state
            # 注意：state_tuple 也可以用于其他目的，比如检查是否到达宝藏，但对于 AI 输入是局部视图

            episode_reward += reward
            steps_this_episode += 1

            # 周期性保存模型和缓冲区
            if agent.total_steps % 5000 == 0 and agent.total_steps > 0: # 例如每 5000 总步数保存一次
                 print(f"Saving model and buffer at total steps {agent.total_steps}")
                 agent.save_model(DQN_WEIGHTS_FILENAME, DQN_BUFFER_FILENAME)


        # 回合结束后的日志记录
        episode_rewards.append(episode_reward)

        # Epsilon 在 agent.get_action() 中根据 total_steps 计算和打印，learn() 中更新 total_steps
        # 这里打印回合结束时的总步数、缓冲区大小、epsilon 值
        print(f"Episode {episode+1}/{num_episodes}: Total Reward = {episode_reward}, Steps = {steps_this_episode}, Epsilon = {agent.epsilon:.4f}, Buffer Size = {len(agent.replay_buffer)}")


        # 每隔一定回合保存模型和缓冲区
        if (episode + 1) % 50 == 0: # 例如每 50 回合保存一次
            print(f"Saving model and buffer at episode {episode+1}")
            agent.save_model(DQN_WEIGHTS_FILENAME, DQN_BUFFER_FILENAME)
            # 保存训练记录 (周期性)
            save_training_record(episode_rewards, DQN_TRAINING_RECORD_FILENAME)


    # 训练结束时保存模型、缓冲区和训练记录
    print("Training finished. Final save.")
    agent.save_model(DQN_WEIGHTS_FILENAME, DQN_BUFFER_FILENAME)
    save_training_record(episode_rewards, DQN_TRAINING_RECORD_FILENAME)

    return episode_rewards


# --- END: DQN Training Function ---


# --- START: Main Execution Block (程序入口) ---
if __name__ == "__main__":
    print("Maze Treasure Hunt AI Script (DQN)")
    print("Choose an option:")
    print("1: Run Pygame Visualization (Trained DQN Agent or Random)")
    print("2: Run DQN Training")
    print("Any other key: Exit")

    choice = input("Enter choice (1 or 2): ")

    # 定义要使用的迷宫尺寸 (概念上的单元格数量)
    MAZE_HEIGHT_CELLS = 10 # 概念高度
    MAZE_WIDTH_CELLS = 10 # 概念宽度
    # 定义一个整数种子可以生成固定迷宫，None 表示每次随机
    # 如果你想在同一个迷宫上训练和可视化，务必设置同一个非 None 的种子
    MAZE_RANDOM_SEED = None # 例如: 42

    # 根据选择运行可视化或训练
    if choice == '1':
        # 运行 Pygame 可视化
        print("\nStarting Pygame visualization...")
        # 生成迷宫布局
        current_maze_layout = generate_maze(MAZE_HEIGHT_CELLS, MAZE_WIDTH_CELLS, random_seed=MAZE_RANDOM_SEED)

        # 定义一个临时的 DQN 代理参数，只用于构建模型以便加载权重
        # 注意：这些参数应与训练时的参数一致，特别是 state_shape
        viz_agent_params = {
            "state_shape": (VIEW_SIZE, VIEW_SIZE, 1), # 状态形状 (局部视图尺寸 x 1 通道)
            "num_actions": NUM_ACTIONS,
            "learning_rate": 0.001, # 学习率在可视化时不重要，但构建模型需要
            "discount_factor": 0.99, # 折扣因子在可视化时不重要
            "epsilon_start": 0.0,    # 可视化时不探索
            "epsilon_end": 0.0,      # 可视化时不探索
            "epsilon_decay_steps": 1, # 可视化时不衰减
            "replay_buffer_capacity": 1, # 可视化不需要大缓冲区
            "batch_size": 1,         # 可视化不训练
            "target_update_frequency": 1, # 可视化不更新目标网络
            "random_seed": None # 可视化随机种子
        }
        # 创建 DQN 代理实例 (此时模型权重是随机的，除非加载成功)
        viz_agent = DQNAgent(**viz_agent_params)

        # 尝试加载训练好的模型权重
        model_loaded = viz_agent.load_model(DQN_WEIGHTS_FILENAME, buffer_filename=None) # 可视化不需要加载缓冲区

        if model_loaded:
             print("Using loaded DQN model for visualization.")
             # 将当前生成的迷宫布局和加载了权重的 DQN 代理实例传递给可视化函数
             run_maze_visualization(current_maze_layout, ai_agent=viz_agent)
        else:
             print("DQN model weights not found. Using random actions for visualization.")
             # 没有加载到模型，使用随机行动可视化
             run_maze_visualization(current_maze_layout, ai_agent=None)

        print("\nPygame visualization finished.")

    elif choice == '2':
        # 运行 DQN 训练
        print("\nStarting DQN training...")
        # 生成迷宫布局
        training_maze_layout = generate_maze(MAZE_HEIGHT_CELLS, MAZE_WIDTH_CELLS, random_seed=MAZE_RANDOM_SEED)

        # 调用训练函数
        # train_dqn_agent 函数内部会创建 env 和 agent
        train_rewards = train_dqn_agent(training_maze_layout)
        print("\nTraining Rewards per Episode:")
        # 打印训练函数返回的最终奖励列表
        print(train_rewards)
        print("\nTraining finished.")

    else:
        print("\nExiting script.")
# --- END: Main Execution Block ---
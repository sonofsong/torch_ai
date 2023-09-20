# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 17:31:38 2023

@author: 165046

for Marcov Decision Process(MDP)

solve with Bellman Optimality Equation (state - action function)
                                       Q-learning
with value iterative policy optimizing
"""
import numpy as np

# Initialize Markov Decision Process model
actions = (0, 1)  # actions (0=left, 1=right)
states = (0, 1, 2, 3, 4)  # states (tiles)
rewards = [-1, -1, 10, -1, -1]  # Direct rewards per state
gamma = 0.9  # discount factor
# Transition probabilities per state-action pair
# Terminating state (all probs 0)
probs = [
    [[0.9, 0.1, 0, 0, 0], [0.1, 0.9, 0, 0, 0]],
    [[0.9, 0, 0.1, 0, 0],[0.1, 0, 0.9, 0, 0]],
    [[0, 0, 0, 0, 0],    [0, 0, 0, 0, 0]],        
    [[0, 0, 0.9, 0, 0.1], [0, 0, 0.1, 0, 0.9]],
    [[0, 0, 0, 0.9, 0.1], [0, 0, 0, 0.1, 0.9]]]

# Set value iteration parameters
episode = 30  # Maximum number of episode
pi = [None, None, None, None, None]  # Initialize policy
learning_rate = 0.2

#行動選択 0 : 左、　1 : 右
def get_action(_q_table, _observation):
    epsilon = 0.002
    if np.random.uniform(0, 1) > epsilon:
        _action = np.argmax(_q_table[_observation])
    else:
        _action = np.random.choice([0, 1])
    return _action

# Q-table(State-Action価値)初期化
q_table = np.zeros((5, 2))
# 各パターンのカウント用
checkPoint = np.zeros((5, 2))

# 指定エピソード数反復
for ep in range(episode):
    
    # 学習過程確認用
    if ep % 5 == 0:     
        print(f"State-Action_table # {ep+1}")
        print(f"{q_table}")
        
    #エピソード開始時の初期状態を設定
    status = np.random.choice([0,1,3,4])
    
    # 終了条件の判断は後半で実施する
    while True:
        # ε-グリーディ法で行動を選択
        action = get_action(q_table, status)
        
        # 各状態ー行動パターンのカウントのため
        checkPoint[status][action] += 1
                
        # 環境モデルからの観測結果（次の状態）
        observation = np.random.choice(5,1,p=probs[status][action])[0]
        
        # 次の状態でえられる最大q-table値
        next_max_q_table = np.sort(q_table[observation])[::-1][0]
        
        # State-action table更新式
        q_table[status][action] += learning_rate * \
            (rewards[observation] \
             + gamma * next_max_q_table \
             - q_table[status][action]) 
        
        # エピソード終了条件の確認（status　＝　2で終了）　
        if observation == 2:
            break
        
        # 観測した次の状態を
        status = observation
        
# 行動戦略算出
for i in range(5):
    pi[i] = np.argmax(q_table[i])

print(f"\nState-Action table # {ep+1}")
print(f"{q_table}")

print("\nStatus-Action count")
print(f"{checkPoint}")

print("\npi")
print(f"{pi}")



# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:48:40 2023

@author: 165046
"""

import gymnasium as gym
import numpy as np
import time
import pandas as pd
import os

def get_status(_observation):
    env_low = env.observation_space.low # 位置と速度の最小値
    env_high = env.observation_space.high #　位置と速度の最大値
    env_dx = (env_high - env_low) / 20 # 40等分
    # 0〜39の離散値に変換する
    position = int((_observation[0] - env_low[0])/env_dx[0])
    velocity = int((_observation[1] - env_low[1])/env_dx[1])
    return position, velocity


def get_action(_q_table, _observation):
    epsilon = 0.002
    if np.random.uniform(0, 1) > epsilon:
        position, velocity = get_status(_observation)
        _action = np.argmax(_q_table[position][velocity])
    else:
        _action = np.random.choice([0, 1, 2])
    return _action


def update_q_table(_q_table, _action,  _observation, _next_observation, _reward):
    alpha = 0.2     # 学習率    
    gamma = 0.99    # 割引き率 

    # 行動後の状態で得られる最大行動価値 Q(s',a')
    next_position, next_velocity = get_status(_next_observation)
    next_max_q_value = max(_q_table[next_position][next_velocity])

    # 行動前の状態の行動価値 Q(s,a)
    position, velocity = get_status(_observation)
    q_value = _q_table[position][velocity][_action]

    # 行動価値関数の更新
    _q_table[position][velocity][_action] = q_value + alpha * (_reward + gamma * next_max_q_value - q_value)

    return _q_table


if __name__ == '__main__':
    #環境構築
    # human : 直接描画, rgb_array : return rgb
    env = gym.make('MountainCar-v0', render_mode='human')
    
    # Qテーブルの初期化
    q_table = np.zeros((20, 20, 3))
    
    rewards = []
    pth = "C:/Users/165046/Desktop/Q-table/20x20_3/"
    
    # 1500エピソードで学習する
    for episode in range(1500):
        os.makedirs(pth+f"{episode}",exist_ok=True)
        action_step =[]
        total_reward = 0
        observation = env.reset()[0]
        start = time.time()
        
        for cnt in range(200): 
            # 5秒ごとに表示
            if cnt % 199 == 0:
                env.render()
            # ε-グリーディ法で行動を選択
            action = get_action(q_table, observation)
            
            # 行動を保存
            action_step.append(action)
            
            # 車を動かし、観測結果・報酬・ゲーム終了FLG・詳細情報を取得
            next_observation, reward, done, turncate, info = env.step(action)
            

            # Qテーブルの更新
            q_table = update_q_table(q_table, action, observation, next_observation, reward)
            total_reward += reward
            observation = next_observation
            
            # doneがTrueになったら１エピソード終了
            if done:
                #GOALを通過したら記録
                #最高記録の場合Gifも保存
                if total_reward%200 != 0 :
                    print(f'episode: {episode}, total_reward: {total_reward}')
                    pd.DataFrame(action_step).to_csv(f"{pth}/action_stp_{total_reward}_{episode}.csv")
                    
                rewards.append(total_reward)
                break
        if cnt == 199 and done == False:
            rewards.append(-200)

        end = time.time()-start
        if (episode%100 == 0):
            print(f"{episode}ep cost : {end} sec")
        
        #見やすく保存するために次元置き換え
        q_table = np.transpose(q_table, (2,0,1))
        for i in range(3):
            pd.DataFrame(q_table[i]).to_csv(f"{pth}/{episode}/q_table_{i}.csv")
        #元の順番に戻せるための次元置き換え
        q_table = np.transpose(q_table, (1,2,0))
        
    pd.DataFrame(rewards).to_csv(f"{pth}/reward_40_40.csv")
    env.close()
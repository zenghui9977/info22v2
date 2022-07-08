import numpy as np
import math
import random
from utils import random_pick


def D_UCB(K, ud, task, u, h, choosetime, f, args):
    # a1 = 0.1 # 移动成本系数
    distance = [0] * args.num_packets
    last = []

    True_reward = [0] * args.num_packets
    ESTIMATED_Q = [0] * args.num_packets

    SCORE = [0] * args.num_packets
    q = 0.5  # 衰减因子
    b1 = 0.3  # CPU周期转换系数
    k = 0
    cost = [0] * args.num_packets
    e = 2.71
    choose_action = 0
    top_k = 8
    F = args.CR

    # Step 1. 對每個红包的平均期望值與置信區間更新
    for i in range(0, args.num_packets):
        cost[i] = args.move_cost_coef * math.exp((math.sqrt((task[i]['L'][0] - ud[u][0]) ** 2 + (task[i]['L'][1] - ud[u][1]) ** 2)))
        True_reward[i] = task[i]['Money'] / h[task[i]['gridnumber']] - cost[i]
        SCORE[i] = task[i]['Money'] / h[task[i]['gridnumber']] - cost[i]
        #choose_action = np.argmax(True_reward)   # 根据At选择行动
        #更新紀錄器，给出被选择的红包序号
        ESTIMATED_Q[i] = q ** (F - f) * ESTIMATED_Q[i] + (
                SCORE[i] - ESTIMATED_Q[i]) /(f + 1)
        choosetime[task[i]['gridnumber']] += q ** (F - f)
        #CHOOSE_TIMES[choose_action] += q ** (N - n)
        True_reward[i] = ESTIMATED_Q[i] + 2 * max(SCORE) * np.sqrt(
            args.delta * np.log(F) / choosetime[task[i]['gridnumber']])


    top_k_reward_index = np.array(True_reward).argsort()[::-1][0:top_k]
    # top_k_reward = [True_reward[i] for i in top_k_reward_index]
    top_k_reward = []
    for i in top_k_reward_index:
        if SCORE[i] < 0:
            top_k_reward.append(0.01)
        else:
            top_k_reward.append(SCORE[i])
    prob_choose = [item/sum(top_k_reward) for item in top_k_reward]
    
    choose_action = random_pick(top_k_reward_index, prob_choose)
    # print(top_k_reward_index, top_k_reward, prob_choose, choose_action)

    # print([task[item]['gridnumber'] for item in top_k_reward_index])
    

    # else:
    for i in range(0, args.num_packets):
        distance[i] = math.sqrt((task[i]['L'][0]-ud[u][0]) ** 2+(task[i]['L'][1]-ud[u][1]) ** 2)
    while k < K[u]:
        min_d = min(distance)
        min_d_i = distance.index(min_d)
        distance[distance.index(min_d)] = 10000
        k = k+1
        if (min_d_i!= choose_action):
            last.append(min_d_i)
    last.append(choose_action)
    choose_action_gridnumber = task[choose_action]['gridnumber']
    
    return choose_action, choose_action_gridnumber, cost[choose_action], last
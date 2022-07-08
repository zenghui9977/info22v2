import numpy as np
import math

def greedy(I_max, K, ud, task_greedy, u):
    a1 = 0.1 # 移动成本系数
    a2 = 1  # 能量系数
    distance = [0]*I_max
    last = []
    lastgridnunber=[]
    True_reward = [0] * I_max
    ESTIMATED_Q = [0] * I_max
    CHOOSE_TIMES = [0] * I_max
    SCORE = [0] * I_max
    q = 0.5  # 衰减因子
    b1 = 0.3  # CPU周期转换系数
    k = 0
    bit = 3
    Tmax = 10
    cost = [0]*I_max
    e = 2.71
    # Step 1. 對每個红包的平均期望值與置信區間更新
    for i in range(0, I_max):
        cost[i] = a1 * e**(math.sqrt((task_greedy[i]['L'][0] - ud[u][0]) ** 2 + (task_greedy[i]['L'][1] - ud[u][1]) ** 2))
        True_reward[i] = 1/cost[i]
    choose_action = np.argmax(True_reward)  # 根据At选择行动
    for i in range(0, I_max):
        distance[i] = math.sqrt((task_greedy[i]['L'][0]-ud[u][0]) ** 2+(task_greedy[i]['L'][1]-ud[u][1]) ** 2)
    while k < K[u]:
        min_d = min(distance)
        min_d_i = distance.index(min_d)
        distance[distance.index(min_d)] = 10000
        k = k+1
        if (min_d_i!= choose_action):
            last.append(min_d_i)
    last.append(choose_action)
    choose_action_gridnumber=task_greedy[choose_action]['gridnumber']
    return choose_action, choose_action_gridnumber, cost[choose_action], last
import numpy as np
import math
import random
import Maximum_entropy  #用贪婪算法按熵最大的目的，把每个区域内的红包数量分好
import DUCB    #单个用户的决策：输入是红包任务，输出是决策(选择哪个红包任务)+K个匿名任务保护隐私
import greedy


# M = 10  # M*M个网格
# print("网格数：10*10=", M * M)
# U = 10  # 用户数目
# print("参与的用户总数目：", U)
# I_max = 8 #云准备生成和下发的红包总数
# print("云发放的红包总数：", I_max)
# h = [1]*M*M   #整个网格的热力值（热力图）
# f = [random.randint(2, 10) for j in range(0, U)]  # 每个用户的计算能力
# ud = [[int(random.uniform(0, 9)) for j in range(1, 3)] for i in range(0, U)]
# print(ud)
'''现在设定为16个网格, 所以（0,3）, 后面网格数变化了需要改变'''
# v = [random.uniform(72, 660) for j in range(0, U)]
# '''速度这块只影响到是否超出任务停留时间，不影响其他： 72m/min是步行速度，660m/min为40km/h, 车在市区的移动速度'''
# E = [random.uniform(10, 50) for j in range(0, U)]
# '''可用的能量'''
# K = [int(random.uniform(0, 4)) for j in range(0, U)]  # 选择匿名K个的数量
# print("K匿名数：", K)
# p = [int(random.uniform(0, 3)) for j in range(0, U)]  # 计算功率
# T = [1] * M * M  # 初始化每个人物的停留时间
def local_greedy(I_max, h_greedy, ud, K, U, task_greedy):
    reward = [0] * U  # 这块要添加奖金的计算
    greedy_user = [0] * U
    Choose_greedy = [0] * U
    choosegreedy_action_gridnumber = [0] * U
    greedy_COST = [0] * U
    for u in range(0, U):
        (Choose_greedy[u], choosegreedy_action_gridnumber[u], greedy_COST[u], greedy_user[u]) = greedy.greedy(I_max, K, ud, task_greedy, u)

        '''以上部分是每个用户给出各自的决策和K匿名的干扰任务选项，
        下面将是云中心根据这些选择来划分奖金：
        1.计算出被用户们选中的任务们的冲突值
        '''
    # print("每个用户选中的红包:", Choose, "选择红包的成本:", COST, "用户上报给云的任务:", user)
    '''--------------根据上报的任务，更新任务的冲突值--------------------'''
    for u in range(0, U):
        for i in range(0, len(greedy_user[u])):
            task_greedy[greedy_user[u][i]]['collison'] = task_greedy[greedy_user[u][i]]['collison'] + 1 / len(greedy_user[u])
    '''----------------下面为云中心计算给每个用户的真实报酬-------------------'''
    bonus = [0] * U
    TRUE_reward = [0] * U
    for u in range(0, U):
        for j in range(0, len(greedy_user[u])):
            bonus[u] = bonus[u] + task_greedy[greedy_user[u][j]]['Money'] / (task_greedy[greedy_user[u][j]]['collison']-1)
            #T[task[greedy_user[u][j]]['gridnumber']] = T[task[greedy_user[u][j]]['gridnumber']] + 1
            h_greedy[task_greedy[greedy_user[u][j]]['gridnumber']] = h_greedy[task_greedy[greedy_user[u][j]]['gridnumber']] + 1 / (K[u] + 1)
        reward[u] = bonus[u] / (K[u] + 1) ** 2
        TRUE_reward[u] = int(reward[u] - greedy_COST[u])
    # print("用户选择的任务：", Choose_greedy, "被选中的任务的网格号：", choosegreedy_action_gridnumber, "用户成本：", greedy_COST, "用户收益：", TRUE_reward, "用户平均收益：", sum(TRUE_reward) / len(TRUE_reward))
    return(choosegreedy_action_gridnumber, sum(TRUE_reward) / len(TRUE_reward))
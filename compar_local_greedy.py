import matplotlib.pyplot as plt
import numpy as np
import math
import random
import Maximum_entropy  #用贪婪算法按熵最大的目的，把每个区域内的红包数量分好
import DUCB    #单个用户的决策：输入是红包任务，输出是决策(选择哪个红包任务)+K个匿名任务保护隐私
import torch
from local_greedy import local_greedy
from task_generation import task
from active_sensing_decision import core
from hyper_parameters import arg_parser

# '''最重要的与贪婪地对比实验：横坐标FL迭代次数，纵坐标FL精度'''

# M = 10  # M*M个网格
# print("网格数：#*#=", M * M)
# U = 25  # 用户数目
# #ud = [[int(random.uniform(0, 2)) for j in range(1, 3)] for i in range(0, U)]
# print("参与的用户总数目：", U)
# I_max = 20 #云准备生成和下发的红包总数

# #print("云发放的红包总数：", I_max)
# #h = [1]*M*M   #整个网格的热力值（热力图）


# print("K匿名数：", K)
# h = [1] * M * M
# choosetime=[1]*M*M
# h_greedy = [1] * M * M
# T = [1] * M * M  # 初始化每个人物的停留时间
# T_greedy = [1] * M * M
# F = 30
# decision = [0]*F
# decision_greedy = [0]*F
# ave_reward = [0]*F
# ave_reward_greedy = [0]*F
# #deta = [0.1] * U
# N = 30
# deta=5

args = arg_parser()

h = [1] * args.width_of_grids * args.width_of_grids
h_greedy = [1] * args.width_of_grids * args.width_of_grids

choosetime = [1] * args.width_of_grids * args.width_of_grids

decision = [0] * args.CR
decision_greedy = [0] * args.CR
ave_reward = [0] * args.CR
ave_reward_greedy = [0] * args.CR

K = [int(random.uniform(0, 1)) for j in range(0, args.clients_num)]  # 选择匿名K个的数量

for f in range(args.CR):
    ud = [[int(random.uniform(0, args.width_of_grids)) for j in range(1, 3)] for i in range(0, args.clients_num)]
    g_task = task(args.num_packets, args.width_of_grids, h, choosetime)
    decision[f], ave_reward[f] = core(h, ud, K, g_task, choosetime, f, args)
    print(decision[f])
    task_greedy = task(args.num_packets, args.width_of_grids, h_greedy, choosetime)
    decision_greedy[f], ave_reward_greedy[f] = local_greedy(args.num_packets, h_greedy, ud, K, args.clients_num, task_greedy)
    # print(decision_greedy[f])

print("our F论后平均收益为")
print(ave_reward)
print(sum(ave_reward)/args.CR)
print("greedy F论后平均收益为")
print(ave_reward_greedy)
print(sum(ave_reward_greedy)/args.CR)

    # print(22222222222222222222)
    # print(taskgreedy)
    # print(decision[f])
    # decision_greedy[f] = local_greedy.local_greedy(I_max, ud, K, U, h, task)
'''归一化处理如下：'''


'''绘图如下：'''
# fig = plt.figure()
# x = [0.2, 0.4, 0.8, 1.6, 3.2]
# y1 = Ave_guiyi_reward[0]
# y2 = Ave_guiyi_reward[1]
# y3 = Ave_guiyi_reward[2]
# plt.plot(x, y1, color='r', linestyle='-')
# plt.plot(x, y2, color='b', linestyle='dashdot')
# plt.plot(x, y3, color='g', linestyle='solid')
# plt.show()
"兰》红》绿； 其实也可以加个横坐标是迭代轮次，纵坐标是平均收益;图：看随着迭代次数的增多，不同探索值的影响"
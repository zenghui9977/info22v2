import numpy as np
import math
import random
import Maximum_entropy  #用贪婪算法按熵最大的目的，把每个区域内的红包数量分好
def task(I_max, M, h, choosetime):
    if (int(M*M/I_max)==50):
        Are = 2
    if (int(M*M/I_max)==25):
        Are = 4
    if (int(M*M/I_max)!=25 and int(M*M/I_max)!=50):
        Are = 10  # 划分的区域数：要与总网格数可以整除
    Area = [0] * Are  # 区域
    b1 = 1000
    '''红包金额设置的系数：b1一个是热力值系数，b2一个是与任务停留时间相关的系数'''
    b2 = 100
    a = Maximum_entropy.entropy_greedy(Area, Are, I_max, h, M)
    '上面完成了红包位置的选择，下面是红包任务的生成'
    task = []  # 初始化任务集合
    money = [0] * len(a[2])  # 初始化每个任务的金额



    col = [1] * I_max  # 初始化每个任务的冲突值

    required_image_number = 10  # 每个任务需要采集的图片数量
    '''下面为红包任务的生成：'''
    for i in range(0, len(a[2])):
        # money[i] = b1/((sum(h)/(M*M))-h[a[2][i]]+1)
        money[i] = b1 / h[a[2][i]] + b2*(np.mean(h) - h[a[2][i]])
        task.append({'L': a[3][i],
                     'Money': money[i],
                     'access_Time': choosetime[a[2][i]],
                     'collison': int(col[i]),
                     'req': required_image_number, 'gridnumber': a[2][i]})

    '''------------------------------------------------------------------:以上部分完成红包任务的位置选择和生成，由云中心完成'''
    return task
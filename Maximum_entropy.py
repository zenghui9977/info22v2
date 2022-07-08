import numpy as np
import math
import random
'''
    N: 分成N个大区域
    I_max：需要投放的红包总数
    h:  热力值
    '''
def entropy_greedy(Area, Are, I_max, h, M):
    I = set()
    for p in range(0, I_max):
        # 对每个红包进行遍历
        L = [0]*Are
        for i in range(0, len(Area)):
            # 为每个红包遍历位置
            nn = [0]*Are
            nn[i] = 1
            HSD = []
            for j in range(0, len(Area)):
                if nn[j] > 0 or Area[j] > 0:
                    HSD.append(- (Area[j] + nn[j]) / I_max * math.log((Area[j] + nn[j]) / I_max, 2))
            L[i] = sum(HSD)
        p_selected_i = L.index(max(L))  # 求被选中的区域编号
        Area[p_selected_i] = Area[p_selected_i] + 1  # 求被选中区域的红包数
        I.add(p_selected_i)
    selected = []
    coordinate = []
    select=[]
    for area in range(0, Are):
        H = []
        for i in range(int(area * M * M / Are), int(area * M * M / Are + M * M / Are)):
            H.append(1 / h[i])
        for j in range(0, Area[area]):
            selectedH = max(H)
            selectedi = H.index(selectedH) + int(area * M * M / Are)
            H[H.index(selectedH)] = 0
            selected.append(selectedi)
    for j in range(0, len(selected)):
        coordinate.append([selected[j] // M, selected[j] % M])
    # print("区域号：", I, "每个区域洒落的红包数目：", Area, "选定的放置红包的网格号：", selected, "选定的放置红包的具体位置：", coordinate)
    return(I, Area, selected, coordinate)


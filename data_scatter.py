import numpy as np
from numpy import random
from numpy.core.fromnumeric import size
import torch
import collections

from torch.utils import data
from torchvision import datasets
from utils import get_data
from hyper_parameters import arg_parser


def split_data_by_label(dataset):
    # 将数据依据label分开，分成一个一个桶
    if torch.is_tensor(dataset.targets):
        labels = dataset.targets.numpy()
    else:
        labels = np.array(dataset.targets)

    labels = np.array(labels)
    
    # print(labels)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    mnist_counter = collections.Counter(idxs_labels[1, :])
    # print(mnist_counter)

    label_num = len(mnist_counter)

    label_idxs_buckets = []
    pos_temp = 0
    for i in range(label_num):
        label_idxs_buckets.append(list(idxs[pos_temp: pos_temp + mnist_counter[i]]))
        pos_temp += mnist_counter[i]
        # print(len(label_idxs_buckets[i]))
    return label_idxs_buckets

def distribute_label_table(label_num_list, user_num):
    label_table = []
    label_num = len(label_num_list)
    for i in label_num_list:
        temp = []
        for j in reversed(label_num_list):
            temp.append([i, j])
        if len(temp) < user_num//label_num:
            for k in range(user_num//label_num - len(temp)):
                temp.append(temp[k])
        label_table.append(temp)
    # print(label_table)
    label_table = [item for sublist in label_table for item in sublist]
    # print(label_table)
    return label_table       

def iid_equal(dataset, num_grids):
    '''
    dataset: the raw dataset read from torchvision
    '''
    num_items = int(len(dataset) / num_grids)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_grids):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def noniid_equal(dataset, num_grids, batch_size):
    dict_users = {i: np.array([]) for i in range(num_grids)}
    shards_size = len(dataset) // (2 * num_grids)

    label_buckets = split_data_by_label(dataset)

    # print(label_buckets)

    label_table = distribute_label_table(range(len(label_buckets)), num_grids)


    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)

    # 每个user先分配一部分数据
    for i in range(num_grids):
        user_label = label_table[i]

        for u_l in user_label:
            if shards_size >= len(label_buckets[u_l]):
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[u_l]), shards_size, replace=False)
                label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
            dict_users[i] = np.concatenate((dict_users[i], list(rand_set)),axis=0).astype(int)

    # 把剩下的进行分配
    for i in range(len(label_buckets)):

        if len(label_buckets[i]) != 0:
            rest_size = len(label_buckets[i]) - len(label_buckets[i]) % batch_size
            rest_set = np.random.choice(list(label_buckets[i]), rest_size, replace=False)
            dict_users[i * i] = np.concatenate((dict_users[i], list(rest_set)),axis=0).astype(int)

    return dict_users


def area_distribution(dataset, num_grids, area_num):
    dict_users = {i: np.array([]) for i in range(num_grids)}
    shards_size = len(dataset) // (2 * num_grids)
    label_buckets = split_data_by_label(dataset)
    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)

    area_label_list = np.arange(label_num)
    np.random.shuffle(area_label_list)
    area_label_list = np.array_split(area_label_list, area_num)

    width_of_grids = int(np.sqrt(num_grids))
    area_point_pos = [[0, 0], [0, width_of_grids//2], [width_of_grids//2, 0], [width_of_grids//2, width_of_grids//2]]
    # print(np.array(area_point_pos[1]) + np.array(area_point_pos[2]))
   
    for area in range(area_num):
        label_in_area = area_label_list[area]
        for index in range(int(num_grids//area_num)):
            peace_of_label_size = shards_size // len(label_in_area)
            for u_l in label_in_area:
                if peace_of_label_size >= len(label_buckets[u_l]):
                    rand_set = np.random.choice(list(label_buckets[u_l]), peace_of_label_size, replace=True)
                else:
                    rand_set = np.random.choice(list(label_buckets[u_l]), peace_of_label_size, replace=False)
                    label_buckets[u_l] = label_buckets[u_l] - set(rand_set)
                pos_temp = (area_point_pos[area][0] + int(index/(width_of_grids//2))) * width_of_grids + area_point_pos[area][1] + index % (width_of_grids//2)

                dict_users[pos_temp] = np.concatenate((dict_users[pos_temp], list(rand_set)),axis=0).astype(int)

    
    # print(dict_users)
    return dict_users


def distribute_part_data_to_clients(dataset, clients_num, grids_num, part_of_data):
    dict_clients = {i: np.array([]) for i in range(clients_num)}
    dict_grids = {i: np.array([]) for i in range(grids_num)}

    label_buckets = split_data_by_label(dataset)
    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)

    label_list = np.arange(label_num)
    np.random.shuffle(label_list)
    seperate_index = int(part_of_data * label_num)
    clients_label_list = label_list[:seperate_index]

    # clients_datasize = [len(label_buckets[i]) for i in clients_label_list]
    # grids_datasize = [len(label_buckets[i]) for i in grids_label_list]

    # size_client, size_grid = sum(clients_datasize) // (clients_num * len(clients_label_list)), sum(grids_datasize) // (grids_num * len(grids_label_list))

    size_client = 400
    size_client_shards = size_client // len(clients_label_list)

    size_grid = (len(dataset) - size_client * clients_num) // grids_num
    size_grid_shard = size_grid // len(label_list)

    # 给clients分配
    for c in range(clients_num):
        for c_l in clients_label_list:
            if size_client_shards >= len(label_buckets[c_l]):
                rand_set = np.random.choice(list(label_buckets[c_l]), size_client_shards, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[c_l]), size_client_shards, replace=False)
                label_buckets[c_l] = label_buckets[c_l] - set(rand_set)
            dict_clients[c] = np.concatenate((dict_clients[c], list(rand_set)),axis=0).astype(int)
     
    # 给网格分配
    for g in range(grids_num):
        for g_l in label_list:
            if size_grid_shard >= len(label_buckets[g_l]):
                rand_set = np.random.choice(list(label_buckets[g_l]), size_grid_shard, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[g_l]), size_grid_shard, replace=False)
                label_buckets[g_l] = label_buckets[g_l] - set(rand_set)
            dict_grids[g] = np.concatenate((dict_grids[g], list(rand_set)),axis=0).astype(int)
    
    return dict_clients, dict_grids
   
def distribute_part_data_to_clients_area(dataset, clients_num, grids_num, part_of_data, area_num):
    dict_clients = {i: np.array([]) for i in range(clients_num)}
    dict_grids = {i: np.array([]) for i in range(grids_num)}

    label_buckets = split_data_by_label(dataset)
    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)

    label_list = np.arange(label_num)
    np.random.shuffle(label_list)
    seperate_index = int(part_of_data * label_num)
    clients_label_list = label_list[:seperate_index]

    # clients_datasize = [len(label_buckets[i]) for i in clients_label_list]
    # grids_datasize = [len(label_buckets[i]) for i in grids_label_list]

    # size_client, size_grid = sum(clients_datasize) // (clients_num * len(clients_label_list)), sum(grids_datasize) // (grids_num * len(grids_label_list))
    # size_grid = size_grid * area_num
    size_client = 400
    size_client_shards = size_client // len(clients_label_list)

    size_grid = (len(dataset) - size_client * clients_num) // grids_num
    

    # 给clients分配
    for c in range(clients_num):
        for c_l in clients_label_list:
            if size_client_shards >= len(label_buckets[c_l]):
                rand_set = np.random.choice(list(label_buckets[c_l]), size_client_shards, replace=True)
            else:
                rand_set = np.random.choice(list(label_buckets[c_l]), size_client_shards, replace=False)
                label_buckets[c_l] = label_buckets[c_l] - set(rand_set)
            dict_clients[c] = np.concatenate((dict_clients[c], list(rand_set)),axis=0).astype(int)
     
    np.random.shuffle(label_list)
    area_label_list = np.array_split(label_list, area_num)

    width_of_grids = int(np.sqrt(grids_num))
    area_point_pos = [[0, 0], [0, width_of_grids//2], [width_of_grids//2, 0], [width_of_grids//2, width_of_grids//2]]

    for area in range(area_num):
        label_in_area = area_label_list[area]
        size_grid_shard = size_grid // len(label_in_area)
        for index in range(int(grids_num//area_num)):
            # peace_of_label_size = size_grid // len(label_in_area)
            # print(size_grid)
            for g_l in label_in_area:
                if size_grid >= len(label_buckets[g_l]):
                    rand_set = np.random.choice(list(label_buckets[g_l]), size_grid_shard, replace=True)
                else:
                    rand_set = np.random.choice(list(label_buckets[g_l]), size_grid_shard, replace=False)
                    label_buckets[g_l] = label_buckets[g_l] - set(rand_set)
                pos_temp = (area_point_pos[area][0] + int(index/(width_of_grids//2))) * width_of_grids + area_point_pos[area][1] + index % (width_of_grids//2)
                dict_grids[pos_temp] = np.concatenate((dict_grids[pos_temp], list(rand_set)),axis=0).astype(int)


    return dict_clients, dict_grids

def generate_label_appear_time(basic_time, rare_ratio, label_num, grids_num):
    label_appear_time_in_grids = [basic_time] * int(rare_ratio * label_num) + [int(grids_num - basic_time * rare_ratio * label_num)//int((1 - rare_ratio) * label_num)] * int((1 - rare_ratio) * label_num)
    for _ in range(grids_num - sum(label_appear_time_in_grids)):
        add_index = np.random.randint(label_num)
        label_appear_time_in_grids[add_index] = label_appear_time_in_grids[add_index] + 1
    return label_appear_time_in_grids

def generate_clients_list_in_each_label(clients_num, common_label_num):
    clients_num_list_in_each_label = [clients_num//common_label_num] * common_label_num
    for _ in range(clients_num - sum(clients_num_list_in_each_label)):
        add_index = np.random.randint(common_label_num)
        clients_num_list_in_each_label[add_index] += 1
    return clients_num_list_in_each_label


def distribute_data_into_grids(dataset, grids_num, clients_num, rare_ratio):
    dict_clients = {i: np.array([]) for i in range(clients_num)}
    dict_grids = {i: np.array([]) for i in range(grids_num)}

    label_buckets = split_data_by_label(dataset)
    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)

    data_size_in_each_grid = len(dataset) // grids_num
    basic_label_scatter_time = 5

    label_appear_time_in_grids = generate_label_appear_time(basic_label_scatter_time, rare_ratio, label_num, grids_num)
    np.random.shuffle(label_appear_time_in_grids)
    grids_num_set = set(np.arange(grids_num))
    
    label_scatter_pos_list = []

    # grids distribution
    for label in range(label_num):
        label_scatter_pos = np.random.choice(list(grids_num_set), label_appear_time_in_grids[label], replace=False)
        label_scatter_pos_list.append(list(label_scatter_pos))
        # print(label_scatter_pos)
        grids_num_set = grids_num_set - set(label_scatter_pos)
        for pos in label_scatter_pos:
            rand_set = np.random.choice(list(label_buckets[label]), data_size_in_each_grid, replace=False)
            dict_grids[pos] = np.concatenate((dict_grids[pos], list(rand_set)),axis=0).astype(int)

    
    # client distribution
    common_label_index = np.array(label_appear_time_in_grids).argsort()[::-1][:int((1 - rare_ratio) * label_num)]
    clients_num_in_each_common_label = generate_clients_list_in_each_label(clients_num, len(common_label_index))

    client_i = 0
    for i in range(len(common_label_index)):
        clients_pos = np.random.choice(list(label_scatter_pos_list[common_label_index[i]]), clients_num_in_each_common_label[i], replace=True) 
        for pos_index in range(clients_num_in_each_common_label[i]):     
            dict_clients[client_i] = dict_grids[clients_pos[pos_index]]
            client_i += 1

    return dict_grids, dict_clients

def distribute_data_into_grids_ba(dataset, grids_num, clients_num, rare_ratio, datasize_grids_num, basic_appear_time):
    dict_clients = {i: np.array([]) for i in range(clients_num)}
    dict_grids = {i: np.array([]) for i in range(grids_num)}
    
    label_buckets = split_data_by_label(dataset)
    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)

    basic_label_scatter_time = basic_appear_time
    label_appear_time_in_grids = generate_label_appear_time(basic_label_scatter_time, rare_ratio, label_num, grids_num)
    np.random.shuffle(label_appear_time_in_grids)

    grids_num_set = set(np.arange(grids_num))

    common_label_index = list(np.array(label_appear_time_in_grids).argsort()[::-1][:int((1 - rare_ratio) * label_num)])
    rare_label_index = list(set(range(label_num)) - set(common_label_index))

    # print(label_appear_time_in_grids)
    # print(common_label_index)
    # print(rare_label_index)
    
    data_size_in_each_grid = datasize_grids_num

    label_scatter_pos_list = []
    clients_scatter_pos_list = []

    # grids distribution
    for label in range(label_num):
        
        label_scatter_pos = np.random.choice(list(grids_num_set), label_appear_time_in_grids[label], replace=False)
        label_scatter_pos_list.append(list(label_scatter_pos))
        grids_num_set = grids_num_set - set(label_scatter_pos)
    print(label_scatter_pos_list)


    rare_label_frag_size = data_size_in_each_grid // (len(rare_label_index))

    for r_label in rare_label_index:
        r_pos = label_scatter_pos_list[r_label]
        for pos in r_pos:
            for dis_label in rare_label_index:
                rand_set = np.random.choice(list(label_buckets[r_label]), rare_label_frag_size, replace=False)
                dict_grids[pos] = np.concatenate((dict_grids[pos], list(rand_set)),axis=0).astype(int)
    
    common_label_main_size = data_size_in_each_grid // 2
    common_label_frag_size = data_size_in_each_grid // (2 * (len(common_label_index) - 1)) 
    # print(common_label_frag_size)
    for c_label in common_label_index:
        c_pos = label_scatter_pos_list[c_label]
        for pos in c_pos:
            for dis_label in common_label_index:
                if dis_label == c_label:
                    rand_set = np.random.choice(list(label_buckets[dis_label]), common_label_main_size, replace=False)
                else:
                    rand_set = np.random.choice(list(label_buckets[dis_label]), common_label_frag_size, replace=False)
                dict_grids[pos] = np.concatenate((dict_grids[pos], list(rand_set)),axis=0).astype(int)
    
    print(label_scatter_pos_list)
    print(common_label_index)
    clients_num_in_each_common_label = generate_clients_list_in_each_label(clients_num, len(common_label_index))
    print(clients_num_in_each_common_label)
    # clients distribution
    client_i = 0
    for i in range(len(common_label_index)):
        clients_pos = np.random.choice(list(label_scatter_pos_list[common_label_index[i]]), clients_num_in_each_common_label[i], replace=False) 
        clients_scatter_pos_list.append(list(clients_pos))
        for pos_index in range(clients_num_in_each_common_label[i]):     
            dict_clients[client_i] = dict_grids[clients_pos[pos_index]]
            client_i += 1
    # print(clients_scatter_pos_list)
    common_label_scatter_pos_list = [item for i in common_label_index for item in label_scatter_pos_list[i]]
    rare_label_scatter_pos_list = [item for i in rare_label_index for item in label_scatter_pos_list[i]]
    print(common_label_scatter_pos_list)
    print(rare_label_scatter_pos_list)

    return dict_grids, dict_clients, clients_scatter_pos_list, common_label_scatter_pos_list, rare_label_scatter_pos_list


def distribute_data_for_cifar100(dataset, grids_num, clients_num, rare_ratio, datasize_grids_num, mix_size):
    dict_clients = {i: np.array([]) for i in range(clients_num)}
    dict_grids = {i: np.array([]) for i in range(grids_num)}   

    label_buckets = split_data_by_label(dataset)
    label_buckets = [set(i) for i in label_buckets]
    label_num = len(label_buckets)
    label_list = range(label_num)
    grids_list = range(grids_num)

    # print(label_num)
    # print([len(label_buckets[i]) for i in range(label_num)])

    rare_labels_num = int(label_num * rare_ratio)
    rare_labels_list = np.random.choice(label_list, rare_labels_num, replace=False)
    common_labels_list = list(set(label_list) - set(rare_labels_list))

    rare_grids_num = int(grids_num * rare_ratio)
    rare_grids_list = np.random.choice(grids_list, rare_grids_num, replace=False)
    common_grids_list = list(set(grids_list) - set(rare_grids_list))

    # print(rare_grids_list)
    # print(common_grids_list)
    
    basic_appear_time = grids_num // label_num
    rare_grids_list_for_label = rare_grids_list.reshape((-1, basic_appear_time))
    # print(rare_grids_list_for_label)
    # print(rare_labels_list)

    grid_rare_datasize_each_label = datasize_grids_num // mix_size

    for i in range(rare_labels_num):
        r_label = rare_labels_list[i]
        r_pos_list = rare_grids_list_for_label[i]
        for pos in r_pos_list:
            exist_label_list = np.random.choice(rare_labels_list, mix_size, replace=False)
            for e_label in exist_label_list:
                rand_set = np.random.choice(list(label_buckets[e_label]), grid_rare_datasize_each_label, replace=False)
                dict_grids[pos] = np.concatenate((dict_grids[pos], list(rand_set)),axis=0).astype(int)
    
    # print([len(dict_grids[i]) for i in range(grids_num)])

    grids_common_datasize_each_label = (datasize_grids_num // len(common_labels_list)) + 1
    for c_index in common_grids_list:
        for c_label in common_labels_list:
            rand_set = np.random.choice(list(label_buckets[c_label]), grids_common_datasize_each_label, replace=False)
            dict_grids[c_index] = np.concatenate((dict_grids[c_index], list(rand_set)),axis=0).astype(int)


    # print([len(dict_grids[i]) for i in range(grids_num)])
    
    clients_pos_list = np.random.choice(common_grids_list, clients_num, replace=False)
    # print(clients_pos_list)
    for i in range(clients_num):
        dict_clients[i] = dict_grids[clients_pos_list[i]]

    # print([len(dict_clients[i]) for i in range(clients_num)])

    return dict_grids, dict_clients, clients_pos_list, common_grids_list, rare_grids_list

def distribute_data_for_gtsrb(dataset, grids_num, clients_num, rare_ratio, datasize_grids_num, mix_size):
    dict_clients = {i: np.array([]) for i in range(clients_num)}
    dict_grids = {i: np.array([]) for i in range(grids_num)} 
    label_buckets = split_data_by_label(dataset)
    label_num = len(label_buckets)
    label_index_list = range(label_num)
    grids_index_list = range(grids_num)

    label_len_list = np.array([len(label_buckets[i]) for i in range(len(label_buckets))])
    label_len_sort_index = label_len_list.argsort()

    # print(len(dataset))
    # print(label_len_sort_index)
    # print(label_len_list[label_len_sort_index])

    rare_label_num = int(rare_ratio * label_num)
    # print(rare_label_num)

    rare_label_list = label_len_sort_index[:rare_label_num]
    common_label_list = list(set(label_index_list) - set(rare_label_list))
    # print(rare_label_list)
    # print(common_label_list)

    # print(label_len_list[rare_label_list])

    rare_grids_num = int(rare_ratio * grids_num)
    common_grids_num = grids_num - rare_grids_num

    rare_grids_index_list = np.random.choice(grids_index_list, rare_grids_num, replace=False)
    common_grids_index_list = list(set(grids_index_list) - set(rare_grids_index_list))

    # print(rare_grids_index_list)
    # print(common_grids_index_list)

    grids_common_datasize_each_label = datasize_grids_num // len(common_label_list) + 1
    # distribute the common labels into grids
    for c_g in common_grids_index_list:
        for c_l in common_label_list:
            rand_set = np.random.choice(list(label_buckets[c_l]), grids_common_datasize_each_label, replace=False)
            dict_grids[c_g] = np.concatenate((dict_grids[c_g], list(rand_set)),axis=0).astype(int)
    
    # print(dict_grids)
    # random the clients in some pos and hold the data distributed in the grids
    clients_pos_list = np.random.choice(common_grids_index_list, clients_num, replace=False)
    for i in range(clients_num):
        dict_clients[i] = dict_grids[clients_pos_list[i]]
    # print(dict_clients)
    # rare label distributed in the rare grids
    mix_rare_label_grids_num = rare_grids_num % rare_label_num
    one_rare_label_grids_num = rare_grids_num - mix_rare_label_grids_num

    mix_datasize_each_label = datasize_grids_num // rare_label_num + 1

    # print(mix_rare_label_grids_num, one_rare_label_grids_num)
    # for i in range(rare_grids_num):
    #     r_g = rare_grids_index_list[i]
    #     # 前几个格子混合数据
    #     if i < mix_rare_label_grids_num:
    #         for r_l in rare_label_list:
    #             rand_set = np.random.choice(list(label_buckets[r_l]), mix_datasize_each_label, replace=False)
    #             dict_grids[r_g] = np.concatenate((dict_grids[r_g], list(rand_set)),axis=0).astype(int)
            
    #     else:
    #         r_l = rare_label_list[int((i - mix_rare_label_grids_num) % rare_label_num)]
    #         rand_set = np.random.choice(list(label_buckets[r_l]), datasize_grids_num, replace=True)
    #         dict_grids[r_g] = np.concatenate((dict_grids[r_g], list(rand_set)),axis=0).astype(int)

    for r_g in rare_grids_index_list:
        for r_l in rare_label_list:
            rand_set = np.random.choice(list(label_buckets[r_l]), mix_datasize_each_label, replace=True)
            dict_grids[r_g] = np.concatenate((dict_grids[r_g], list(rand_set)),axis=0).astype(int)



    # print([len(dict_grids[i]) for i in range(grids_num)])
    return dict_grids, dict_clients, clients_pos_list, common_grids_index_list, rare_grids_index_list

args = arg_parser()
trainset, testset = get_data(args)
# distribute_data_for_cifar100(trainset, 400, 20, 0.3, 500, 2)
# dict_clients, dict_grids = distribute_part_data_to_clients_area(trainset, 20, 100, 0.2, 4)
# print(dict_clients, dict_grids)
# print([len(dict_clients[i]) for i in range(len(dict_clients))])
# print([len(dict_grids[i]) for i in range(len(dict_grids))])    
# dict_grids, dict_clients = distrbute_data_into_grids(trainset, 100, 20, 0.4)
# dict_grids, dict_clients, y, _, _ = distribute_data_into_grids_ba(trainset, 200, 20, 0.5, 500, 1)

# print([len(dict_grids[i]) for i in range(100)])
# print([len(dict_clients[i]) for i in range(20)])

distribute_data_for_gtsrb(trainset, 100, 20, 0.4, 500, 10)



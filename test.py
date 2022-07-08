
from centralized_learning import one_epoch_centralize_learning
from hyper_parameters import arg_parser
import datetime
import torch
import sys
import os
import numpy as np
from utils import Logger, get_data, init_model, list_split, print_param
from fl_functions import one_round_federated_learning
from data_scatter import distribute_data_into_grids_ba
from torchvision import datasets, transforms


# from visdom import Visdom
# viz = Visdom()
# viz.line([[0, 0]],[0], win='test', opts=dict(title='accuracy', legend=['our_reward', 'greedy_reward']))


# y_1 = range(20)
# y_2 = [2 * ccc for ccc in range(20)]
# for i in range(20):
#     viz.line([[y_1[i], y_2[i]]], [i], win='test', update='append')

# load the hyper parameter 
args = arg_parser()


# mkdir log file folder
log_dir = './logs/'
sencond_log_dir = '/test/'
if not os.path.exists(log_dir + sencond_log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    os.mkdir(log_dir + sencond_log_dir)

# # log the print info during the exp
sys.stdout = Logger(log_dir + sencond_log_dir + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log')

print_param(args)

# get the dataset
# trainset, testset = get_data(args)
l_data = datasets.Kitti('/data/', train=True, download=True)
img, smnt = l_data[0]
print(img, smnt)

# if torch.is_tensor(trainset.targets):
#     trainlabelset = trainset.targets.numpy()
# else:
#     trainlabelset = np.array(trainset.targets)

# if torch.is_tensor(testset.targets):
#     testlabelset = testset.targets.numpy()
# else:
#     testlabelset = np.array(testset.targets)


# print(trainlabelset[:20])
# print(testlabelset[:20])

# model = init_model(args)
# model = model.cuda()
# model.train()


# grids_num = args.width_of_grids * args.width_of_grids
# dict_grids, dict_clients, clients_location = distribute_data_into_grids_ba(trainset, grids_num, args.clients_num, args.rare_ratio)
# clients_index_list = range(args.clients_num)
# paricipants_in_traditional_fl = int(args.q * args.clients_num)

# print('Centralized')
# for iters in range(args.CR):
#     weight, _, _ = one_epoch_centralize_learning(trainset, testset, range(len(trainset)), model, args)
#     model.load_state_dict(weight)

# print('FL')
# for iters in range(args.CR):
#     chosen_clients = np.random.choice(clients_index_list, paricipants_in_traditional_fl, replace=False)
#     chosen_clients_data_indexs = [dict_clients[i] for i in chosen_clients]
#     weight, _, _ = one_round_federated_learning(trainset, testset, chosen_clients_data_indexs, model, args)
#     model.load_state_dict(weight)


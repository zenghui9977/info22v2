from local_greedy import local_greedy
from active_sensing_decision import core
from task_generation import task
from centralized_learning import one_epoch_centralize_learning
from fl_functions import one_round_federated_learning
import numpy as np
from data_scatter import distribute_data_for_cifar100, distribute_data_for_gtsrb, distribute_data_into_grids, distribute_data_into_grids_ba
import os
import sys
import time
import copy
import torch
import datetime

from torch.utils.data import DataLoader
from hyper_parameters import arg_parser
from utils import Logger, change_index_2_loc, collect_new_data_from_grids, get_data, inference, init_model, print_param, save_info

from visdom import Visdom
viz = Visdom()
viz.line([[0, 0, 0, 0]],[0], win='test_accuracy', opts=dict(title='test_accuracy', legend=['our', 'tra_fl', 'fl+mobile', 'centralized']))
# the graph of training loss
viz.line([[0, 0, 0, 0]],[0], win='tra_accuracy', opts=dict(title='tra_accuracy', legend=['our', 'tra_fl', 'fl+mobile', 'centralized']))


# load the hyper parameter 
args = arg_parser()

# mkdir log file folder
log_dir = './logs/'
sencond_log_dir = '/exp1/'
if not os.path.exists(log_dir + sencond_log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    os.mkdir(log_dir + sencond_log_dir)
log_record_time = str(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
# log the print info during the exp
sys.stdout = Logger(log_dir + sencond_log_dir + log_record_time + '.log')

print_param(args)

# get the dataset
trainset, testset = get_data(args)


global_init_model = init_model(args)
# init the model
traditional_fl_model = copy.deepcopy(global_init_model)
centralized_model = copy.deepcopy(global_init_model)
our_model = copy.deepcopy(global_init_model)
mobile_fl_model = copy.deepcopy(global_init_model)

traditional_fl_model = traditional_fl_model.cuda()
traditional_fl_model.train()

centralized_model = centralized_model.cuda()
centralized_model.train()

our_model = our_model.cuda()
our_model.train()

mobile_fl_model = mobile_fl_model.cuda()
mobile_fl_model.train()

grids_num = args.width_of_grids * args.width_of_grids
if args.dataset == 'cifar100':
    dict_grids, dict_clients, clients_location, common_label_scatter_pos_list, rare_label_scatter_pos_list = distribute_data_for_cifar100(trainset, grids_num, args.clients_num, args.rare_ratio, args.rare_grids_data_num, args.mixed_size)
elif args.dataset == 'gtsrb':
    dict_grids, dict_clients, clients_location, common_label_scatter_pos_list, rare_label_scatter_pos_list = distribute_data_for_gtsrb(trainset, grids_num, args.clients_num, args.rare_ratio, args.rare_grids_data_num, args.mixed_size)
else:
    dict_grids, dict_clients, clients_location, common_label_scatter_pos_list, rare_label_scatter_pos_list = distribute_data_into_grids_ba(trainset, grids_num, args.clients_num, args.rare_ratio, args.rare_grids_data_num, args.basic_appear_tiem)
    clients_location = [i for item in clients_location for i in item]

# get the location of each clients
clients_location = change_index_2_loc(clients_location, args.width_of_grids)

# our setttings
h = [1] * args.width_of_grids * args.width_of_grids 
choosetime = [1] * args.width_of_grids * args.width_of_grids
K = [int(np.random.uniform(0, args.K_ano)) for j in range(args.clients_num)]
our_dict_clients = copy.deepcopy(dict_clients)


# traditional fl settings
paricipants_in_traditional_fl = int(args.q * args.clients_num)
clients_index_list = range(args.clients_num)
tra_fl_dict_clients = copy.deepcopy(dict_clients)

# FL + mobile
fl_mobile_dict_clients = copy.deepcopy(dict_clients)
mobile_clients_num = int(args.clients_num * args.mobile_client_ratio)
rara_mobile_clients_num = max(int(args.rare_mobile_client_ratio * mobile_clients_num), 0)
com_mobile_clients_num = mobile_clients_num - rara_mobile_clients_num
mobile_dict_clients = copy.deepcopy(dict_clients)

# records
our_accuracy_list, traditional_accuracy_list, mobile_fl_accuracy_list, centralized_accuracy_list = [], [], [], []
our_tra_accuracy_list, traditional_tra_accuracy_list, mobile_fl_tra_accuracy_list, centralized_tra_accuracy_list = [0], [], [], []

for cr in range(args.CR):
    print('|' + '-' * 10 + str(cr) + '-' * 10 + '|')
    # choose the clients will participate in the training process
    chosen_clients = np.random.choice(clients_index_list, paricipants_in_traditional_fl, replace=False)

    
    print('|' + '-' * 10 + 'our' + '-' * 10 + '|')
    task_in_cr = task(args.num_packets, args.width_of_grids, h, choosetime)
    decision_in_cr, rewards_in_cr = core(h, clients_location, K, task_in_cr, choosetime, cr, args)
    print(decision_in_cr)
    # The clients collect the grids data
    our_dict_clients = collect_new_data_from_grids(our_dict_clients, dict_grids, range(args.clients_num), decision_in_cr)
    our_chosen_clients_data_indexs = [our_dict_clients[i] for i in chosen_clients]
    our_weight, our_tra_acc, our_tes_acc = one_round_federated_learning(trainset, testset, our_chosen_clients_data_indexs, model_=copy.deepcopy(our_model), args=args)
    if our_tra_acc < (our_tra_accuracy_list[-1] - 0.2):
        print('load previous model weight')
        our_model.load_state_dict(our_model.state_dict())
        our_tra_acc, _ = inference(our_model, trainset)
        our_tes_acc, _ = inference(our_model, testset)
    else:
        print('load new model weight')
        our_model.load_state_dict(our_weight)

    print('|' + '-' * 10 + 'traditional FL' + '-' * 10 + '|')   
    chosen_clients_data_indexs = [tra_fl_dict_clients[i] for i in chosen_clients]
    traditional_fl_weight, traditional_fl_tra_acc, traditional_fl_tes_acc = one_round_federated_learning(trainset, testset, chosen_clients_data_indexs, model_=copy.deepcopy(traditional_fl_model), args=args)
    traditional_fl_model.load_state_dict(traditional_fl_weight)
    

    print('|' + '-' * 10 + 'FL + mobile' + '-' * 10 + '|')
    moving_client_index = np.random.choice(clients_index_list, mobile_clients_num, replace=False)
    # clients move to common grids
    moving_to_com_decision = np.random.choice(common_label_scatter_pos_list, com_mobile_clients_num, replace=False)
    mobile_dict_clients = collect_new_data_from_grids(mobile_dict_clients, dict_grids, moving_client_index[:com_mobile_clients_num], moving_to_com_decision)
    # clients move to some grids with rare label
    moving_to_rar_decision = np.random.choice(rare_label_scatter_pos_list, rara_mobile_clients_num, replace=False)
    mobile_dict_clients = collect_new_data_from_grids(mobile_dict_clients, dict_grids, moving_client_index[com_mobile_clients_num:], moving_to_rar_decision)
    mobile_chosen_clients_data_indexs = [mobile_dict_clients[i] for i in chosen_clients]
    mobile_weight, mobile_tra_acc, mobile_tes_acc = one_round_federated_learning(trainset, testset, mobile_chosen_clients_data_indexs, model_=copy.deepcopy(mobile_fl_model), args=args)
    mobile_fl_model.load_state_dict(mobile_weight)


    print('|' + '-' * 10 + 'centralized learning' + '-' * 10 + '|')
    centralized_chosen_clients_data_indexs = [temp for i in chosen_clients for temp in our_dict_clients[i]]
    centralized_weight, centralized_tra_acc, centralized_tes_acc = one_epoch_centralize_learning(trainset, testset, centralized_chosen_clients_data_indexs, model_c=copy.deepcopy(centralized_model), args=args)
    centralized_model.load_state_dict(centralized_weight)


    our_accuracy_list.append(our_tes_acc)
    traditional_accuracy_list.append(traditional_fl_tes_acc)
    mobile_fl_accuracy_list.append(mobile_tes_acc)
    centralized_accuracy_list.append(centralized_tes_acc)

    our_tra_accuracy_list.append(our_tra_acc)
    traditional_tra_accuracy_list.append(traditional_fl_tra_acc)
    mobile_fl_tra_accuracy_list.append(mobile_tra_acc)
    centralized_tra_accuracy_list.append(centralized_tra_acc)

    viz.line([[our_tra_acc, traditional_fl_tra_acc, mobile_tra_acc, centralized_tra_acc]], [cr], win='tra_accuracy', update='append')
    viz.line([[our_tes_acc, traditional_fl_tes_acc, mobile_tes_acc, centralized_tes_acc]], [cr], win='test_accuracy', update='append')
    time.sleep(0.001)


result_folder = './result/'

name_list = ['our_accuracy', 'traditional_accuracy', 'mobile_fl_accuracy', 'centralized_accuracy', 'our_tra_accuracy', 'traditional_tra_accuracy', 'mobile_fl_tra_accuracy', 'centralized_tra_accuracy']
result_list = [our_accuracy_list, traditional_accuracy_list, mobile_fl_accuracy_list, centralized_accuracy_list, our_tra_accuracy_list, traditional_tra_accuracy_list, mobile_fl_tra_accuracy_list, centralized_tra_accuracy_list]
file_name = 'exp1_' + str(args.dataset) + log_record_time +'.csv'
save_info(result_list, name_list, result_folder, file_name)


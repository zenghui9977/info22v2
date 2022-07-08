import os
import sys
import copy
import time
import datetime
import numpy as np
from numpy.lib.function_base import append
from hyper_parameters import arg_parser
from active_sensing_decision import core
from local_greedy import local_greedy
from task_generation import task
from fl_functions import one_round_federated_learning
from data_scatter import distribute_data_into_grids_ba, distribute_data_for_cifar100, distribute_data_for_gtsrb
from utils import Logger, inference, print_param, get_data, init_model, collect_new_data_from_grids, save_info, change_index_2_loc


from visdom import Visdom
viz = Visdom()
viz.line([[0, 0]],[0], win='accuracy', opts=dict(title='accuracy', legend=['test_accuracy', 'train_accuracy']))
# the graph of training loss
viz.line([[0, 0]],[0], win='rewards', opts=dict(title='rewards', legend=['rewards', 'greedy_rewards']))


# load the hyper parameter 
args = arg_parser()

# mkdir log file folder
log_dir = './logs/'
sencond_log_dir = '/exp2/'
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

our_model = copy.deepcopy(global_init_model)

our_model = our_model.cuda()
our_model.train()


grids_num = args.width_of_grids * args.width_of_grids
if args.dataset == 'cifar100':
    dict_grids, dict_clients, clients_location, common_label_scatter_pos_list, rare_label_scatter_pos_list = distribute_data_for_cifar100(trainset, grids_num, args.clients_num, args.rare_ratio, args.rare_grids_data_num, args.mixed_size)
elif args.dataset == 'gtsrb':
    dict_grids, dict_clients, clients_location, common_label_scatter_pos_list, rare_label_scatter_pos_list = distribute_data_for_gtsrb(trainset, grids_num, args.clients_num, args.rare_ratio, args.rare_grids_data_num, args.mixed_size)
else:
    dict_grids, dict_clients, clients_location, common_label_scatter_pos_list, rare_label_scatter_pos_list = distribute_data_into_grids_ba(trainset, grids_num, args.clients_num, args.rare_ratio, args.rare_grids_data_num, args.basic_appear_tiem)
    clients_location = [i for item in clients_location for i in item]

clients_location = change_index_2_loc(clients_location, args.width_of_grids)

# our setttings
h = [1] * args.width_of_grids * args.width_of_grids 
choosetime = [1] * args.width_of_grids * args.width_of_grids
K = [int(np.random.uniform(0, args.K_ano)) for j in range(args.clients_num)]
our_dict_clients = copy.deepcopy(dict_clients)


# greedy settings
h_greedy = [1] * args.width_of_grids * args.width_of_grids 


clients_index_list = range(args.clients_num)
paricipants_in_traditional_fl = int(args.q * args.clients_num)

our_accuracy_list, our_tra_accuracy_list = [0], [0]
rewards, greedy_reward = [], []

for cr in range(args.CR):
    print('|' + '-' * 10 + str(cr) + '-' * 10 + '|')
    chosen_clients = np.random.choice(clients_index_list, paricipants_in_traditional_fl, replace=False)
    
    print('|' + '-' * 10 + 'our' + '-' * 10 + '|')
    task_in_cr = task(args.num_packets, args.width_of_grids, h, choosetime)
    decision_in_cr, rewards_in_cr = core(h, clients_location, K, task_in_cr, choosetime, cr, args)

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

    task_greedy = task(args.num_packets, args.width_of_grids, h_greedy, choosetime)
    decision_greedy_in_cr, ave_reward_greedy_in_cr = local_greedy(args.num_packets, h_greedy, clients_location, K, args.clients_num, task_greedy)
   
    our_accuracy_list.append(our_tes_acc)
    our_tra_accuracy_list.append(our_tra_acc)
    rewards.append(rewards_in_cr)
    greedy_reward.append(ave_reward_greedy_in_cr)

    viz.line([[our_tes_acc, our_tra_acc]], [cr], win='accuracy', update='append')
    viz.line([[rewards_in_cr, ave_reward_greedy_in_cr]], [cr], win='rewards', update='append')
    time.sleep(0.001)

avg_reward = np.mean(rewards)
avg_greedy_reward = np.mean(greedy_reward)
print('The avg rewards:' + '\t' + str(avg_reward))
print('The avg greedy_reward:' + '\t' + str(avg_greedy_reward))

result_folder = './result/'

name_list = ['our_accuracy', 'our_tra_accuracy', 'rewards', 'greedy_reward', 'avg_reward', 'avg_greedy_reward']
result_list = [our_accuracy_list, our_tra_accuracy_list, rewards, greedy_reward, [avg_reward], [avg_greedy_reward]]
file_name = 'exp2_' + str(args.dataset) + '_' + str(args.num_packets) + '_' + log_record_time + '.csv'
save_info(result_list, name_list, result_folder, file_name)


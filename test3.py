from centralized_learning import one_epoch_centralize_learning
from hyper_parameters import arg_parser
import os
import sys
import datetime
import numpy as np
from utils import Logger, print_param, get_data, init_model, save_info

from visdom import Visdom
viz = Visdom()
viz.line([[0]],[0], win='test_accuracy', opts=dict(title='test_accuracy', legend=['centralized']))
# the graph of training loss
viz.line([[0]],[0], win='tra_accuracy', opts=dict(title='tra_accuracy', legend=['centralized']))


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

trainset, testset = get_data(args)

model = init_model(args)
model = model.cuda()
model.train()

data_index = []
datasize = len(trainset)
step_each_size = 10000
cycle_time = datasize // step_each_size
limited_cycle= 10
tra_accuracy, test_accuracy = [], []
data_index_tmp = []


print('Centralized')
for iters in range(100):
    if((iters + 1)/cycle_time <= 10):
        if (iters + 1) % cycle_time == 0:
            data_index = np.concatenate((data_index, range(datasize)), axis=0).astype(int)
            data_index_tmp = data_index
            print(len(data_index_tmp))
        else:
            data_index_tmp = np.concatenate((data_index, range(((iters + 1)%cycle_time) * step_each_size)), axis=0).astype(int)
    weight, tra_acc, tes_acc = one_epoch_centralize_learning(trainset, testset, data_index_tmp, model, args)
    model.load_state_dict(weight)
    tra_accuracy.append(tra_acc)
    test_accuracy.append(tes_acc)
    viz.line([[tra_acc]], [iters], win='tra_accuracy', update='append')
    viz.line([[tes_acc]], [iters], win='test_accuracy', update='append')


# for iters in range(args.CR):
#     sample_data_index = np.random.choice(data_index, datasize, replace=False)
#     weight, tra_acc, tes_acc = one_epoch_centralize_learning(trainset, testset, sample_data_index, model, args)
#     model.load_state_dict(weight)
#     tra_accuracy.append(tra_acc)
#     test_accuracy.append(tes_acc)
#     viz.line([[tra_acc]], [iters], win='tra_accuracy', update='append')
#     viz.line([[tes_acc]], [iters], win='test_accuracy', update='append')

result_folder = './result/'

name_list = ['centralized_accuracy', 'centralized_tra_accuracy']
result_list = [test_accuracy, tra_accuracy]
file_name = 'test2_' + str(args.dataset) + log_record_time +'.csv'
save_info(result_list, name_list, result_folder, file_name)
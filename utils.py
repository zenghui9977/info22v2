import copy
from Mymodels import CNNMnist, Cifar_model, FMnist_model
import os
import torch
import random
import sys
import csv
import pandas as pd
import numpy as np
import shutil
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Logger(object):
    def __init__(self, log_file_name="Default.log"):
        self.terminal = sys.stdout
        self.log = open(log_file_name, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_data(args):
    data_dir = './data/'
    if args.dataset == 'mnist':
        
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=apply_transform)

    elif args.dataset == 'cifar':
       
        apply_transform = transforms.Compose([

            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=apply_transform)

    elif args.dataset == 'fmnist':
        
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=False, transform=apply_transform)

        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=False, transform=apply_transform)

    elif args.dataset == 'cifar100':
        cifar100_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        cifar100_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
        apply_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar100_mean, cifar100_std)
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=False, transform=apply_transform)

        test_dataset = datasets.CIFAR100(data_dir, train=False, download=False, transform=test_transform)

    elif args.dataset == 'gtsrb':
        apply_transform = transforms.Compose([
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4, 0.4, 0.4], [0.2, 0.2, 0.2])
        ])

        train_dataset = datasets.ImageFolder(data_dir + '/GTSRB/train/', transform=apply_transform)
        test_dataset = datasets.ImageFolder(data_dir + '/GTSRB/test/', transform=apply_transform)
    return train_dataset, test_dataset

def init_model(args):
    if args.dataset == 'mnist':
        global_model = CNNMnist()
    elif args.dataset == 'fmnist':
        global_model = FMnist_model()
    elif args.dataset == 'cifar':
        global_model = Cifar_model(args.model, pre=args.pretrain)
    elif args.dataset == 'cifar100':
        global_model = Cifar_model(args.model, pre=args.pretrain)
    elif args.dataset == 'gtsrb':
        global_model = Cifar_model(args.model, pre=args.pretrain)

    return global_model

def inference(model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    testloader = DataLoader(test_dataset, batch_size=200, shuffle=False)

    for _, (images, labels) in enumerate(testloader):
        images, labels = images.cuda(), labels.cuda()

        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss

def save_info(data_list, name_list, file_dir, file_name):
    if not os.path.exists(file_dir):
        os.mkdir(file_dir)

    data = pd.DataFrame(index=name_list, data=data_list)

    data.to_csv(os.path.join(file_dir, file_name), mode='a')

def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]


def random_pick(some_list, probabilities):
    x=random.uniform(0,1)
    cumulative_probability=0.0
    for item,item_probability in zip(some_list,probabilities):
        cumulative_probability+=item_probability
        if x < cumulative_probability: break
    return item

def change_index_2_loc(client_location_index, grids_width):
    client_location_pos = []
    for i in client_location_index:
        client_location_pos.append([int(i/grids_width), i%grids_width])
    return client_location_pos

def collect_new_data_from_grids(u_dict_clients, u_dict_grids, clients_cor_index, u_decision):
    u_clients_num = len(clients_cor_index)
    clients_tmp = copy.deepcopy(u_dict_clients)
    for c in range(u_clients_num):
        client_ind = clients_cor_index[c]
        client_dec_ind = u_decision[c]
        clients_tmp[client_ind] = np.concatenate((clients_tmp[client_ind], list(u_dict_grids[client_dec_ind])), axis=0).astype(int)
    return clients_tmp

def print_param(args):
    args_temp = vars(args)
    d = max(map(len, args_temp.keys()))
    print('-' * 10 + 'parameters in this exp' + '-' * 10)
    print('name'.ljust(d + 2) + ':\t' + 'value')
    print('-' * 42)
    for key in args_temp.keys():
        print(str(key).ljust(d + 2) + ':\t' + str(args_temp[key]))
    
    print('-' * 42)

def move_file(src_path, dst_path, file_name):
    if not os.path.isfile(src_path + file_name):
        print('%s not exist ! ! !' % (src_path + file_name))
    else:
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        shutil.move(src_path + file_name, dst_path + file_name)
        print('move %s -> %s' % (src_path + file_name, dst_path + file_name))

def readTrafficSignsTestFormSpecialStructure(rootpath, csv_file_name):
    gtFile = open(rootpath + '/' + csv_file_name + '.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    gtReader.__next__()
    file_names, labels = [], []
    for row in gtReader:
        sample_file_name = row[0]
        sample_label = row[-1]
        move_file(rootpath + '/', rootpath + '/' + format(int(sample_label), '05d') + '/', sample_file_name)

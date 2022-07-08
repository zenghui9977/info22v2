import os
import sys
import time
import copy
import torch
import datetime

from fl_functions import DatasetSplit
from torch.utils.data import DataLoader, dataset
from hyper_parameters import arg_parser
from utils import Logger, get_data, inference, init_model, save_info


def one_epoch_centralize_learning(trainset, testset, data_indexs, model_c, args):
    start_time = time.time()
    train_data = DataLoader(DatasetSplit(trainset, data_indexs), batch_size=args.batch_size, shuffle=True)
    # init the criterion and optimizer in the training
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model_c.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    for iters in range(args.local_centralized_ep):
        for _, (images, labels) in enumerate(train_data):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            log_probs = model_c(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
    train_accuracy, _ = inference(model_c, trainset)
    print("train Accuracy: {:.2f}%".format(100 * train_accuracy))
    accuracy, _ = inference(model_c, testset)
    print("test Accuracy: {:.2f}%".format(100 * accuracy))

    end_time = time.time()
    print('The Total Run Time is :{0:0.4f}'.format(end_time - start_time))

    return model_c.state_dict(), train_accuracy, accuracy


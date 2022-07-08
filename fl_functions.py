import time
import torch
import copy
from torch.utils.data import DataLoader, Dataset
from utils import inference

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone(), torch.tensor(label)

def FL_aggregation(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def client_local_training(trainset, data_index, local_model, args):
    
    train_data = DataLoader(DatasetSplit(trainset, data_index), batch_size=args.local_bs, shuffle=True)
    
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for ep in range(args.local_ep):
        for _, (images, labels) in enumerate(train_data):
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            log_probs = local_model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
    # print(inference(local_model, trainset))
    return local_model.state_dict()
    

def one_round_federated_learning(trainset, testset, data_indexs, model_, args):
    start_time = time.time()
    clients_num = len(data_indexs)
    weights = []
    # pre_train_accuracy, _ = inference(model_, testset)
    # pre_weights = copy.deepcopy(model_).state_dict()
    for c in range(clients_num):
        weight = client_local_training(trainset, data_indexs[c], local_model=copy.deepcopy(model_), args=args)
        weights.append(weight)

    updated_weight = FL_aggregation(weights)
    model_.load_state_dict(updated_weight)

    train_accuracy, _ = inference(model_, trainset)
    print("train Accuracy: {:.2f}%".format(100 * train_accuracy))
    accuracy, _ = inference(model_, testset)
    print("test Accuracy: {:.2f}%".format(100 * accuracy))
    # if accuracy - pre_train_accuracy < - 0.2:
    #     print(pre_train_accuracy, accuracy)
    #     updated_weight = pre_weights

    end_time = time.time()
    print('The Total Run Time is :{0:0.4f}'.format(end_time - start_time))

    return updated_weight, train_accuracy, accuracy


import argparse

def arg_parser():
    parser = argparse.ArgumentParser()
    # framework basic setting
    parser.add_argument('--width_of_grids', type=int, default=10, help='The grids in this network')
    parser.add_argument('--clients_num', type=int, default=20, help='The clients in this framework')
    parser.add_argument('--num_packets', type=int, default=10, help='The red packets number in each round')
    parser.add_argument('--K_ano', type=int, default=3, help='The K-ano')
    parser.add_argument('--CR', type=int, default=100, help='the communication round or iteration')
    parser.add_argument('--iter_MA', type=int, default=30, help='the iteration of Multi-Armed Bandit')
    parser.add_argument('--delta', type=int, default=5, help='the parameter in Multi-Armed Bandit')
    parser.add_argument('--q', type=float, default=0.3, help='the sample ratio in FL')
    parser.add_argument('--rare_ratio', type=float, default=0.4, help='the rare label ratio')
    parser.add_argument('--mobile_client_ratio', type=float, default=0.2, help='the rare label ratio')
    parser.add_argument('--rare_mobile_client_ratio', type=float, default=0.1, help='the rare label ratio in mobile')

    parser.add_argument('--rare_grids_data_num', type=int, default=500, help='the data item size in rare grids')
    parser.add_argument('--basic_appear_tiem', type=int, default=5, help='the basic appear time in all grids')
    parser.add_argument('--mixed_size', type=int, default=20, help='the rare label number in each grid')

    # model training parameters
    parser.add_argument('--dataset', type=str, default='cifar', help='the training dataset: cifar, cifar100, gtsrb')
    parser.add_argument('--model', type=str, default='resnet18', help='the deep learning model used in FL')
    parser.add_argument('--batch_size', type=int, default=200, help='the batch size in centralized learning')
    parser.add_argument('--pretrain', type=bool, default=False, help='whether pretrain the global model')

    # local training parameters
    parser.add_argument('--local_ep', type=int, default=3, help='The epoch number in local training')
    parser.add_argument('--local_bs', type=int, default=100, help='The batch size in local training')
    parser.add_argument('--lr', type=float, default=0.1, help='The learning rate in local training')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum of SGD')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='The weight_decay of SGD')

    parser.add_argument('--local_centralized_ep', type=int, default=1, help='The epoch number in centralized training')

    # paramters in active sensing and decision
    parser.add_argument('--move_cost_coef', type=float, default=0.1, help='the coeffient of moving cost')

    # saving the result in the experiments


    args = parser.parse_args()
    return args


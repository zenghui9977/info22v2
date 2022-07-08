source activate
conda activate ml
python exp2_effectiveness.py --clients_num 20 --num_packets 10 --dataset gtsrb --width_of_grids 10 --rare_grids_data_num 300
python exp2_effectiveness.py --clients_num 20 --num_packets 5 --dataset gtsrb --width_of_grids 10 --rare_grids_data_num 300
python exp2_effectiveness.py --clients_num 20 --num_packets 20 --dataset gtsrb --width_of_grids 10 --rare_grids_data_num 300
python exp2_effectiveness.py --clients_num 20 --num_packets 30 --dataset gtsrb --width_of_grids 10 --rare_grids_data_num 300
python exp2_effectiveness.py --clients_num 20 --num_packets 5 --dataset cifar100
python exp2_effectiveness.py --clients_num 20 --num_packets 20 --dataset cifar100
python exp2_effectiveness.py --clients_num 20 --num_packets 30 --dataset cifar100

# DBDPMNet
DBDPMNet: A Dual-Branch Dynamic Prior Modulation Network for Long-Tailed Recognition

This repository contains the official implementation of DBDPMNet
## Running the Code

### To train DBDPMNet on CIFAR100-LT:
```bash
python3 cifar100_train.py -a resnet32 --dataset cifar100 --loss_type DEBS  --epochs 200 --num_classes 100 --workers 15 --print_freq 50 -b 120 -mixup_prob 1 --gpu 0 --start_data_aug 0 --end_data_aug 20 --lr 0.1 --weighted_alpha 1 --exp_str dbdpmnet_cifar100_lt
```

### To train DBDPMNet on ImageNet-LT:
```bash
python3 imagenet_train.py -a resnet50 --dataset Imagenet-LT --loss_type DEBS  --epochs 200 --num_classes 45 --workers 15 --print_freq 50 -b 120 --mixup_prob 1  --start_data_aug 0 --end_data_aug 20 --lr 0.1 --weighted_alpha 1   --exp dbdpmnet_imagenet_lt
```
### To train DBDPMNet on NWPU-RESISC45-LT:
```bash
python3 imagenet_train.py -a resnet50 --dataset NWPU-LT --loss_type EBS  --epochs 200 --num_classes 45 --workers 15 --print_freq 50 -b 120 --mixup_prob 1  --start_data_aug 0 --end_data_aug 20 --lr 0.1 --weighted_alpha 1   --exp dbdpmnet_nwpu_lt
```

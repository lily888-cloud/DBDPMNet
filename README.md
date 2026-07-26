# DBDPMNet
DBDPMNet: A Dual-Branch Dynamic Prior Modulation Network for Long-Tailed Recognition

This repository contains the official implementation of DBDPMNet
## Running the Code
## Running the Code

### CIFAR100-LT

To train DBDPMNet on CIFAR100-LT:

```bash
python3 cifar100_train.py \
    -a resnet32 \
    --dataset cifar100 \
    --loss_type DEBS \
    --epochs 200 \
    --num_classes 100 \
    --workers 15 \
    --print_freq 50 \
    -b 120 \
    --mixup_prob 1 \
    --gpu 0 \
    --start_data_aug 0 \
    --end_data_aug 20 \
    --lr 0.1 \
    --weighted_alpha 1 \
    --exp_str dbdpmnet_cifar100_lt



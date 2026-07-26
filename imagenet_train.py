# reference code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
import math
import os
import random
from statistics import median
import time
import datetime
import numpy as np
import torch
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from save import prepare_folders,AverageMeter
import torchvision.transforms as transforms
from models import resnet_imagenet_cbam as models 
from tensorboardX import SummaryWriter
from imbalance_data.lt_data import LT_Dataset
from losses import LDAMLoss, BalancedSoftmaxLoss, DynamicEffBalancedSoftmaxLoss
from opts import parser
import warnings
from torch.nn import Parameter
import torch.nn.functional as F
'''from util.util import *
from util.randaugment import rand_augment_transform
import util.moco_loader as moco_loader'''
from numpy import linalg as LA
from sklearn.metrics import confusion_matrix

def set_seeds(seed_value, use_cuda):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False

class TwoHeadNet(nn.Module):
    def __init__(self, arch, num_classes, use_norm):
        super(TwoHeadNet, self).__init__()
        model = models.__dict__[arch](num_classes=num_classes)
        
        self.resnet = model
        num_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()  # Remove the fully connected layer of ResNet

        self.head1 = nn.Sequential(
            nn.Linear(num_features,  num_classes),
        )

        self.head2 = nn.Sequential(
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x1, x2):
        features1 = self.resnet(x1)
        features2 = self.resnet(x2)
        out1 = self.head1(features1)
        out2 = self.head2(features2)

        return out1, out2

best_acc1 = 0

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, args.loss_type, args.train_rule, args.data_aug, str(args.imb_factor),
         str(args.rand_number),
         str(args.mixup_prob), args.exp_str])
    prepare_folders(args)
    if args.cos:
        print("use cosine LR")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 1000
    use_norm = True if args.loss_type == 'LDAM' else False
    model = TwoHeadNet(args.arch, num_classes=num_classes, use_norm=use_norm )
    params1 = list(model.resnet.parameters()) + list(model.head1.parameters())
    params2 = list(model.resnet.parameters()) + list(model.head2.parameters())

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()


    optimizer1 = torch.optim.SGD(params1, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    optimizer2 = torch.optim.SGD(params2, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)    

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.use_randaug:
        print("use randaug!!")
        rgb_mean = (0.485, 0.456, 0.406)
        ra_params = dict(translate_const=int(224 * 0.45),
                         img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        augmentation_randncls = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
            rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params),
            transforms.ToTensor(),
            normalize,
        ]
        augmentation_sim = [
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)  # not strengthened
            ], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco_loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]
        transform_train = [transforms.Compose(augmentation_randncls), transforms.Compose(augmentation_sim)]

        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])

    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_dataset = LT_Dataset(args.root, '/data/ImageNet_LT/train.txt', transform_train,
                               use_randaug=args.use_randaug)
    val_dataset = LT_Dataset(args.root, '/data/ImageNet_LT/test.txt', transform_val)

    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == 1000

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)


    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    weighted_train_loader = None

    if args.data_aug == 'CMeO':
        cls_weight = 1.0 / (np.array(cls_num_list) ** args.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print(samples_weight)
        weighted_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        weighted_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, pin_memory=True,
                                                            sampler=weighted_sampler)

        cls_num_list = np.array(cls_num_list)
        cls_num_list_med = abs(cls_num_list-np.median(cls_num_list))+1 
        cls_weight = 1.0 / ((cls_num_list_med) ** args.weighted_alpha)
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)

        samples_weight = np.array([cls_weight[t] for t in train_dataset.targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print(samples_weight)
        median_sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),
                                                                  replacement=True)
        median_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                            num_workers=args.workers, pin_memory=True,
                                                            sampler=median_sampler)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()

    # init log for training
    # log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    # log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    # with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
    #     f.write(str(args))
    # tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    start_time = time.time()
    print("Training started!")

    for epoch in range(args.start_epoch, args.epochs):
        if args.use_randaug:
            paco_adjust_learning_rate(optimizer1, epoch, args)
            paco_adjust_learning_rate(optimizer2, epoch, args)
        else:
            adjust_learning_rate(optimizer1, epoch, args)
            adjust_learning_rate(optimizer2, epoch, args)

        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'CBReweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 80
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Sample rule is not listed')

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'BS':
            criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args.gpu)
        elif args.loss_type == 'DEBS':
            criterion = DynamicEffBalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn('Loss type is not listed')
            return
        tau = get_tau(epoch, 200, tau_min=0.5,tau_max=1.2)

        # train for one epoch
        train(train_loader, model, criterion, optimizer1, optimizer2,   epoch, args, weighted_train_loader=weighted_train_loader, median_train_loader = median_train_loader,tau=tau)

        # evaluate on validation set
        #acc1 = validate(val_loader, model, criterion, epoch, args)

        acc1, many_acc, med_acc, few_acc = validate(val_loader, model, criterion, epoch, args,tau=tau)

        # remember best acc@1 and save checkpoint
        torch.save(model.state_dict(), os.path.join(args.root_log, args.store_name, "model.pth"))
        '''is_best = acc1 > best_acc1
        if is_best:
            model_final = model
            torch.save(model_final.state_dict(), os.path.join(args.root_log, args.store_name, "model_best.pth"))
        best_acc1 = max(acc1, best_acc1)

        # tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        # log_testing.write(output_best + '\n')
        # log_testing.flush()'''

        is_best = acc1 > best_acc1
        if is_best:
            model_final = model
            model_path = os.path.join(args.root_log, args.store_name, "model_best.pth")
            torch.save(model_final.state_dict(), model_path)

            # === 日志路径 ===
            log_path = os.path.join(args.root_log, args.store_name, "通道-真cbam-tau0.5-1.2.log")

            # === 以追加模式写入，不覆盖旧结果 ===
            with open(log_path, "a") as f:
                f.write("========== Best Model Updated ==========\n")
                f.write("Epoch: {}\n".format(epoch))
                f.write("Overall Acc@1: {:.3f}\n".format(acc1))
                f.write("Many-shot Avg Acc: {:.3f}\n".format(many_acc))
                f.write("Medium-shot Avg Acc: {:.3f}\n".format(med_acc))
                f.write("Few-shot Avg Acc: {:.3f}\n".format(few_acc))
                f.write("Model saved to: {}\n".format(model_path))
                f.write("========================================\n\n")

            print(f"✅ Best model updated at epoch {epoch}, log appended to: {log_path}")

        best_acc1 = max(acc1, best_acc1)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)


        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, epoch + 1)

    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))
    #torch.save(model_final.state_dict(), os.path.join(args.root_log, args.store_name, "model_best.pth"))
    



def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def get_tau(epoch, max_epoch, tau_min=0.5, tau_max=1.2):
    return tau_min + (tau_max - tau_min) * (1 - math.cos(math.pi * epoch / max_epoch)) / 2


def train(train_loader, model, criterion, optimizer1, optimizer2, epoch, args, log=None,
              tf_writer=None, weighted_train_loader=None, median_train_loader=None,tau=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    end = time.time()
    if args.data_aug == 'CMeO':
        weighted_train_loader = iter(weighted_train_loader)
        median_iter = iter(median_train_loader)
        criterion2 = DynamicEffBalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args.gpu)


    for i, (input, target) in enumerate(train_loader):
        if args.data_aug == 'CMeO' and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
            try:
                input2, target2 = next(weighted_train_loader)
                input_m, target_m = next(median_iter)
            except:
                weighted_train_loader = iter(weighted_train_loader)
                input2, target2 = next(weighted_train_loader)
                median_iter = iter(median_train_loader)
                input_m, target_m = next(median_iter)
            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

            input_m = input_m[:input.size()[0]]
            target_m = target_m[:target.size()[0]]
            input_m = input_m.cuda(args.gpu, non_blocking=True)
            target_m = target_m.cuda(args.gpu, non_blocking=True)

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        # Data augmentation
        r = np.random.rand(1)

        if args.data_aug == 'CMeO' and args.start_data_aug < epoch < (
                args.epochs - args.end_data_aug) and r < args.mixup_prob:
            # generate mixed sample

            lam1 = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam1)
            input[:, :, bbx1:bbx2, bby1:bby2] = input_m[:, :, bbx1:bbx2, bby1:bby2]
            lam1 = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            
            
            lam2 = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(input2.size()[0]).cuda()
            target_a = target2
            target_b = target2[rand_index]
            input2 = lam2 * input2 + (1-lam2)* input2[rand_index]
            
            # compute output
            output1, output2 = model(input,input2)

            loss1 = criterion(output1, target,tau) * lam1 + criterion(output1, target_m,tau) * (1. - lam1)
            loss2 = criterion2(output2, target_a,tau) * lam2 + criterion2(output2, target_b,tau) * (1. - lam2)


            #loss = loss1 + loss2

        else:
           inverse_iter = iter(weighted_train_loader)
           input2, target2 = next(inverse_iter)
           input2, target2 = input2.cuda(args.gpu, non_blocking=True), target2.cuda(args.gpu, non_blocking=True)
           output1, output2 = model(input, input2)
           loss1 = criterion(output1, target,tau)
           loss2 = criterion(output2, target2,tau)


        losses.update(loss1.item(), input.size(0))
        
        # compute gradient and do SGD step
        optimizer1.zero_grad()
        optimizer2.zero_grad()

        
        loss = loss1 + loss2
        loss.backward()

        optimizer1.step()
        optimizer2.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer1.param_groups[-1]['lr']))  # TODO
            print(output)

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def rand_bbox_withcenter(size, lam, cx, cy):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val',tau=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    
    '''weight_head = model.head1[0].weight.cpu().detach().numpy()
    weight_tail = model.head2[0].weight.cpu().detach().numpy()'''
    
    if isinstance(model, torch.nn.DataParallel):
        weight_head = model.module.head1[0].weight.cpu().detach().numpy()
        weight_tail = model.module.head2[0].weight.cpu().detach().numpy()
    else:
        weight_head = model.head1[0].weight.cpu().detach().numpy()
        weight_tail = model.head2[0].weight.cpu().detach().numpy()

    weight_norm_head = LA.norm(weight_head,axis=1)
    weight_norm_tail = LA.norm(weight_tail,axis=1)
    
    head_scale = np.mean(weight_norm_head)/(np.mean(weight_norm_head)+np.mean(weight_norm_tail) )
    tail_scale = np.mean(weight_norm_tail)/(np.mean(weight_norm_head)+np.mean(weight_norm_tail))
    
    print(tail_scale, head_scale)
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output1, output2 = model(input, input)
            #print(output1.shape, output2.shape)
            
            
            many_shot = train_cls_num_list > 100
            medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list >= 20)
            few_shot = train_cls_num_list < 20

            output1= output1*head_scale
            output2= output2*tail_scale

            output = (output1+output2)/2
            loss = criterion(output, target,tau )
            losses.update(loss.item(), input.size(0))

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)

        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list > 20)
        few_shot = train_cls_num_list <= 20
        print("many avg, med avg, few avg", float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)),
              float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)),
              float(sum(cls_acc[few_shot]) * 100 / sum(few_shot)))


    #return top1.avg
    return top1.avg, float(sum(cls_acc[many_shot]) * 100 / sum(many_shot)), \
           float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot)), \
           float(sum(cls_acc[few_shot]) * 100 / sum(few_shot))


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 80:
        lr = args.lr * 0.01
    elif epoch > 60:
        lr = args.lr * 0.1
    else:
        lr = args.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def paco_adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    warmup_epochs = 10

    lr = args.lr
    if epoch < warmup_epochs:
        lr = lr / warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs + 1) / (args.epochs - warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.lr_steps:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(args, state, is_best, epoch):
    filename = os.path.join(args.root_model, args.store_name, 'checkpoint_{0:03d}.pth.tar'.format(epoch))
    torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.join(args.root_model, args.store_name, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()

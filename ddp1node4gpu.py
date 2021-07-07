"""
Script for Training on One Node, One GPU
Adapted from tutorial: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

Execution Command(s)
$ sudo apt update
$ pip3 install torch
$ pip3 install torchvision


$ pip install -r requirements.txt
$ export MASTER_ADDR=<IP-ADDR-LEADER>
$ export MASTER_PORT=8888
$ python3 ddp1node4gpu.py -n 1 -g 4 -nr 0 --epochs 20
"""

import os
import copy
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import datasets
import time
from torch import optim
from torch.optim import lr_scheduler

DATA_PATH = './data/'
VAL_PATH = './data/val/'
TRAIN_PATH = './data/train/'
IS_SHUFFLED = False  #Set to True for Single GPU. False for Multi-GPU
BATCH_SIZE = 128
NUM_WORKERS = 0

TRANSFORM = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def load_images(path):
    return datasets.ImageFolder(path, TRANSFORM)

def sampler(dataset, gpu, args):      #multi-gpu
    rank = args.nr * args.gpus + gpu
    return torch.utils.data.distributed.DistributedSampler(
    	dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )

def dataloader(dataset, sampler):
    return torch.utils.data.DataLoader(dataset=dataset,
                                        batch_size=BATCH_SIZE,
                                        shuffle=IS_SHUFFLED,
                                        num_workers=NUM_WORKERS,
                                        pin_memory=True,
                                        sampler=sampler)

def launch_group(gpu, args):       #Multi-gpu
    rank = args.nr * args.gpus + gpu	                          
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )


def train(gpu, args):
    start_time = time.time()
    since = time.time()
    torch.manual_seed(0)

    # load model, configure training params
    model = torchvision.models.resnet18(pretrained=True, progress=False)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    learning_rate = 0.0001
    momentum = 0.9
    train_acc= list()
    valid_acc= list()

    # define loss function (criterion) and optimizer and scheduler
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=momentum)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Wrap the model for DDP execution
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])

    # train,val raw images -> sampler -> train,val dataloaders -> dict(train,val)
    train_dataset = load_images(TRAIN_PATH)
    train_sampler = sampler(train_dataset, gpu, args)
    train_loader = dataloader(train_dataset, train_sampler)
    val_dataset = load_images(VAL_PATH)
    val_sampler = sampler(val_dataset, gpu, args)
    val_loader = dataloader(val_dataset, val_sampler)
    phase_dict = {'train': train_loader, 'val': val_loader}
    phase_size = {'train': len(train_dataset), 'val': len(val_dataset)}

    start = datetime.now()
    total_step = len(train_loader)

    for epoch in range(args.epochs):
        print('Epoch {}/{}'.format(epoch + 1, args.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phase_dict.keys():
            is_train_phase = (phase == 'train')
            size = phase_size[phase]
            if is_train_phase:
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_prec = 0.0
            running_rec = 0.0
            running_f1 = 0.0

            for i, (images, labels) in enumerate(phase_dict[phase]):
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # track history if only in train
                with torch.set_grad_enabled(is_train_phase):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if is_train_phase:
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)

                cur_acc = torch.sum(preds == labels.data).double() / BATCH_SIZE
                print("\npreds:", preds)
                print("label:", labels.data)
                print("%d-th epoch, %d-th batch (size=%d), %s acc= %.3f \n" % (
                    epoch+1, i+1, len(labels), phase, cur_acc))

                if is_train_phase:
                    train_acc.append(cur_acc)
                else:
                    valid_acc.append(cur_acc)

                epoch_loss = running_loss / size
                epoch_acc = running_corrects.double() / size

                print('{} Loss: {:.4f} Acc: {:.4f} \n\n'.format(
                phase, epoch_loss, epoch_acc))

                # deep copy the model
                if (not is_train_phase) and (epoch_acc > best_acc):
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc= %.3f at Epoch: %d' % (best_acc, best_epoch))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, train_acc, valid_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes            # Multi-gpu
    mp.spawn(train, nprocs=args.gpus, args=(args,))     # Multi-gpu
    # NOTE: MASTER_ADDR and MASTER_P0RT to be set in terminal as env variables

if __name__ == '__main__':
    main()
"""
Script for Training on One Node, One GPU
Adapted from tutorial: https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

Execution Command(s)
$ sudo apt update
OR sudo yum update
$ pip3 install torch
$ pip3 install torchvision


export PATH=env:// export MASTER_ADDR=3.234.239.60
export PATH=env:// export MASTER_PORT=8888
$ pip3 install -r requirements.txt
$ export MASTER_ADDR=<IP-ADDR-LEADER>
$ export MASTER_PORT=8888

$ ssh -i "Jon-ashley-nodirbek-keypair.pem" ec2-user@

One each node in the cluster...
Master:    $ python3 ddp1node4gpu.py -n 4 -g 4 -nr 0 --epochs 20
Worker 1:  $ python3 ddp1node4gpu.py -n 4 -g 4 -nr 1 --epochs 20
Worker 2:  $ python3 ddp1node4gpu.py -n 4 -g 4 -nr 2 --epochs 20
Worker 3:  $ python3 ddp1node4gpu.py -n 4 -g 4 -nr 3 --epochs 20

Testing 2 nodes:
Master:    $ python3 ddp4node4gpu.py -n 2 -g 1 -nr 0 --epochs 5
Worker 1:  $ python3 ddp4node4gpu.py -n 2 -g 1 -nr 1 --epochs 5

Optional env variables for debugging:
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

To kill all python processes: $ sudo pkill python
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

def train(gpu, args):
    def print_write(s, f):
        print(s)
        f.write(s)

    print('train function entered...\n')
    is_master = False
    if args.nr == 0:
        is_master = True
    f = open('./ddp-train-logs.txt', 'w')

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
        print('loading & transforming images...\n')
        return datasets.ImageFolder(path, TRANSFORM)

    
    def sampler(dataset, gpu, args):      #multi-gpu
        print('configuring sampler ...\n')
        rank = args.nr * args.gpus + gpu
        return torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=rank
        )

    def dataloader(dataset, sampler):
        print('executing dataloader...\n')
        return torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=BATCH_SIZE,
                                            shuffle=IS_SHUFFLED,
                                            num_workers=NUM_WORKERS,
                                            pin_memory=True,
                                            sampler=sampler)

    def launch_group(gpu, args):       #Multi-gpu
        print('launching process groups...\n')
        rank = args.nr * args.gpus + gpu	                          
        dist.init_process_group(                                   
            backend='nccl',                                         
            init_method='tcp://52.3.236.107:8888',  # 'tcp://<master ip addr>:8888'                               
            world_size=args.world_size,                              
            rank=rank                                               
        )

    launch_group(gpu, args)

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
        s = f'Node: {args.nr}\n'
        print_write(s, f)
        s = f'Epoch {epoch + 1}/{args.epochs}\n'
        print_write(s, f)
        s = '' * 10
        print_write(s, f)

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

                s = f'Node: {args.nr}\n'
                print_write(s,f)
                s = f"\npreds: {preds}\n"
                print_write(s, f)
                s = f"label: {labels.data}\n"
                print_write(s, f)
                s = f"{epoch+1}-th epoch, {i+1}-th batch (size={len(labels)}), {phase} acc={cur_acc}\n"
                print_write(s, f)

                if is_train_phase:
                    train_acc.append(cur_acc)
                else:
                    valid_acc.append(cur_acc)

                epoch_loss = running_loss / size
                epoch_acc = running_corrects.double() / size

                s = f'{phase} Loss: {epoch_loss} Acc: {epoch_acc} \n\n'
                print_write(s,f)

                # deep copy the model
                if (not is_train_phase) and (epoch_acc > best_acc):
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since

        s = f'Training complete in {time_elapsed // 60}m {time_elapsed % 60}s'
        print_write(s, f)
        s = f'Best val Acc= {best_acc} at Epoch: {best_epoch}\n'
        print_write(s, f)

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model, './covid_resnet18_epoch%d.pt' %epoch )
    return model, train_acc, valid_acc, f


def main():
    print('ddp4node4gpy.py Running...\n')
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
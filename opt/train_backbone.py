from utils.datasets import MiniImageNet, TieredImageNet, CifarFS

# %%
# Credit: Copied from
# https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py

import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def main(args):
    # the train and evaluation transformations
    train_crop_size = 224

    norm_img_mean = [0.485, 0.456, 0.406]
    norm_img_std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=norm_img_mean, std=norm_img_std)

    train_transforms = transforms.Compose([transforms.RandomResizedCrop(train_crop_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize])
    eval_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          normalize])
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features, device_ids=args.device_ids)
        model.to(args.device)
    else:
        # model = torch.nn.DataParallel(model, device_ids=args.device_ids).to(args.device)
        model.to(args.device)

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint.get('epoch', 0)
            best_prec1 = checkpoint.get('best_prec1', 0)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                raise RuntimeError(f'no state dictionary found in {args.resume}')
            model.load_state_dict(state_dict)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, args.start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.dataset == 'imagenet':
        # Data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(traindir, train_transforms),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, eval_transforms),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    elif args.dataset in ('miniimagenet', 'tieredimagenet', 'cifarfs'):
        data_root = f'{args.data}'
        target_transform = None

        dsconstructor = {'miniimagenet': MiniImageNet,
                         'tieredimagenet': TieredImageNet,
                         'cifarfs': CifarFS}[args.dataset]
        full_trainset = dsconstructor(root=data_root, data_type='base',
                                      transform=train_transforms,
                                      target_transform=target_transform)

        trainset, evalset = full_trainset.split([90, 10])
        trainset.transform = train_transforms
        evalset.transform = eval_transforms

        use_dali = False
        if use_dali:
            # dali can perform the pre-processing on the gpu.
            # If you're in a cpu crunch, you may find it useful.
            # However, you may find its settings/argument names
            # a bit counter-intuitive.
            from utils.datasets import get_dali_loader
            train_files = [x[0] for x in trainset.samples]
            train_labels = [x[1] for x in trainset.samples]
            train_loader = get_dali_loader(train_files, train_labels, batch_size=args.batch_size,
                                           num_workers=8, crop_size=train_crop_size,
                                           random_area=None, norm_mean=norm_img_mean,
                                           norm_std=norm_img_std, shuffle=True)

            eval_files = [x[0] for x in evalset.samples]
            eval_labels = [x[1] for x in evalset.samples]
            val_loader = get_dali_loader(eval_files, eval_labels, batch_size=args.batch_size,
                                         num_workers=8, crop_size=train_crop_size,
                                         random_area=None, norm_mean=norm_img_mean,
                                         norm_std=norm_img_std, shuffle=False)
        else:
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            val_loader = torch.utils.data.DataLoader(
                evalset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

    else:
        raise Exception(f'Unknown dataset {args.dataset}')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(args.device)

    if args.optim_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     betas=(0.9, 0.999), eps=1e-08,
                                     weight_decay=args.weight_decay,
                                     amsgrad=False)
    else:
        raise Exception(f'Unknown optimizer type : {args.optim_type}')

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, lr_decay_rate=args.lr_decay_rate,
                             lr_decay_epochs=args.lr_decay_epochs)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        last_filename = args.store + 'last.pth.tar'
        best_filename = args.store + 'best.pth.tar'
        os.makedirs(os.path.dirname(last_filename), exist_ok=True)
        os.makedirs(os.path.dirname(best_filename), exist_ok=True)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, last_filename=last_filename, best_filename=best_filename)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(device=args.device, non_blocking=False)
        input_var = input.cuda(device=args.device, non_blocking=False)
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        if device_str.startswith('cuda'):
            target = target.cuda(device=args.device, non_blocking=True)
        input_var = input.cuda(device=args.device, non_blocking=False)
        target_var = target

        # compute output
        output = model(input_var)
        # In case a feature vector is also produced.
        if isinstance(output, tuple):
            output = output[0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, last_filename='checkpoint.pth.tar',
                    best_filename='model_best.pth.tar'):
    torch.save(state, last_filename)
    if is_best:
        shutil.copyfile(last_filename, best_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_decay_rate=0.1, lr_decay_epochs=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (lr_decay_rate ** (epoch // lr_decay_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    use_pytorch_example_argparse = False
    if use_pytorch_example_argparse:
        import argparse
        parser = argparse.ArgumentParser(description='PyTorch (Mini)ImageNet Training')
        parser.add_argument('data', metavar='DIR', help='path to dataset')
        parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                            choices=model_names,
                            help='model architecture: ' +
                                 ' | '.join(model_names) +
                                 ' (default: resnet18)')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=90, type=int, metavar='N',
                            help='number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=256, type=int,
                            metavar='N', help='mini-batch size (default: 256)')
        parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                            metavar='LR', help='initial learning rate')
        parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                            help='momentum (only used for sgd)')
        parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                            metavar='W', help='weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=10, type=int,
                            metavar='N', help='print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')
        parser.add_argument('--store', default='', type=str, metavar='PATH',
                            help='prefix path to store latest checkpoint (default: none)')
        parser.add_argument('--device', default='cuda', type=str,
                            help='device to run everything on')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                            help='use pre-trained model')
        parser.add_argument('--optim-type', default='sgd', type=str,
                            help='The optimizer type (sgd or adam)')
        parser.add_argument('--lr-decay-rate', default=0.1, type=float,
                            help='learning rate decay rate')
        parser.add_argument('--lr-decay-epochs', default=30, type=int,
                            help='learning rate decay epochs')
    else:
        use_argparse = True
        if use_argparse:
            import argparse
            my_parser = argparse.ArgumentParser(description='PyTorch Few-Shot Backbone Training')
            my_parser.add_argument('--resnet_no', default=18, type=int, required=True)
            my_parser.add_argument('--dataset', default='miniimagenet', type=str, required=True)
            my_parser.add_argument('--device', default='cuda:0', type=str, required=True)
            my_parser.add_argument('--dataroot', default='', type=str, required=False)
            args_parser = my_parser.parse_args()
            resnet_no = args_parser.resnet_no
            dataset_name = args_parser.dataset
            device_str = args_parser.device
            args_dataroot = args_parser.dataroot
        else:
            resnet_no = 10
            dataset_name = 'cifarfs'
            device_str = 'cuda:0'
            args_dataroot = ''

        assert dataset_name in ('tieredimagenet', 'miniimagenet', 'cifarfs')

        class Args:
            def __init__(self):
                pass

        args = Args()

        train_ver = 2
        args.arch = f'resnet{resnet_no}'
        args.pretrained = False
        args.dataset = dataset_name
        args.data = f'./datasets/{args.dataset}' if (args_dataroot == '') else args_dataroot
        args.resume = f'./backbones/{dataset_name}_resnet{resnet_no}_v{train_ver}_best.pth.tar'
        args.store = f'./backbones/{dataset_name}_resnet{resnet_no}_v{train_ver}_'
        # Will create 'backbones/miniimagenet_resnet18_v2_last.pth.tar' and
        #             'backbones/miniimagenet_resnet18_v2_best.pth.tar' for example.
        args.workers = 8
        args.device = torch.device(device_str)
        args.device_ids = [args.device.index]
        # args.device_ids can be used for distributed training, e.g., args.device_ids = [0, 1, 2, 3]
        args.evaluate = False
        args.print_freq = 10

        if dataset_name == 'miniimagenet':
            # These are the settings from the original pytorch example
            # script and we used them to train the backbone used in our
            # main paper's figures and tables (the mini-imagenet dataset).
            args.start_epoch = 0
            args.epochs = 90
            args.batch_size = 256
            args.optim_type = 'sgd'
            args.lr = 0.1
            args.momentum = 0.9
            args.weight_decay = 3e-4
            args.lr_decay_epochs = 30
            args.lr_decay_rate = 0.1
        elif dataset_name in ('tieredimagenet', 'cifarfs'):
            # For the rebuttals, we were asked to add the tiered-imagenet and
            # the cifar-fs datasets. Due to the time-crunch in the rebuttals, we
            # tried the Adam optimizer and the training settings of the baseline++
            # method of the "A Closer Look at Few-shot Classification" paper.
            args.optim_type = 'adam'
            args.start_epoch = 0
            args.epochs = 400
            args.batch_size = 16
            args.lr = 0.001
            args.momentum = 0.9
            args.weight_decay = 0
            args.lr_decay_epochs = 400
            args.lr_decay_rate = 1.
        else:
            raise ValueError(f'Dataset {dataset_name} not implemented.')

    main(args)

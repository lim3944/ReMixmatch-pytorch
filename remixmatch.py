import argparse
import logging
import math
import os
import random
import shutil
import time
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from torchvision import transforms
import random

from cifar_remixmatch import get_cifar10, get_cifar100
from utils import AverageMeter, accuracy

import wideresnet as models

import pdb

logger = logging.getLogger(__name__)

DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}

best_acc = 0


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def main():
    parser = argparse.ArgumentParser(description='PyTorch ReMixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--epochs', default=500, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--k-img', default=65536, type=int,
                        help='number of labeled examples')
    parser.add_argument('--out', default='experiment/',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar"),
    parser.add_argument("--beta", type=float, default=0.5,
                        help="mixup rate")
                    

    args = parser.parse_args()
    
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        '''
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        '''
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))

        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        if args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 10
        if args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed != -1:
        set_seed(args)

    args.out = args.out + "result_"+str(args.num_labeled)
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        writer = SummaryWriter(args.out)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        '~/htlim/data/'+args.dataset, args.num_labeled, args.k_img, args.k_img * args.mu)

    model = create_model(args)

    '''
    model_rot = models.classifier(in_channel=3, num_classes=4, filters=32)
    optimizer_rot = optim.SGD(model_rot.parameters(), lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    
    model_rot.to(args.device)
    '''
    #multi GPU
    '''
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model_rot = nn.DataParallel(model_rot)
    '''

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.iteration = args.k_img // args.batch_size // args.world_size
    args.total_steps = args.epochs * args.iteration
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup * args.iteration, args.total_steps)

    if args.use_ema:
        ema_model = ModelEMA(args, model, args.ema_decay, device)

    start_epoch = 0

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        #model_rot.load_state_dict(checkpoint['state_dict2'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    test_accs = []


    model.zero_grad()
    #model_rot.zero_grad()

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_loss_x, train_loss_u, train_loss_us, train_loss_r = train(
            args, labeled_trainloader,unlabeled_trainloader,
            model, optimizer, ema_model, scheduler, epoch)
        
        if args.no_progress:
            logger.info("Epoch {}. train_loss: {:.4f}. train_loss_x: {:.4f}. train_loss_u: {:.4f}. train_loss_us: {:.4f}. train_loss_r: {:.4f}."
                        .format(epoch+1, train_loss, train_loss_x, train_loss_u, train_loss_us, train_loss_r))

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        test_loss, test_acc = test(args, test_loader, test_model, epoch)

        if args.local_rank in [-1, 0]:
            writer.add_scalar('train/1.train_loss', train_loss, epoch)
            writer.add_scalar('train/2.train_loss_x', train_loss_x, epoch)
            writer.add_scalar('train/3.train_loss_u', train_loss_u, epoch)
            writer.add_scalar('train/3.train_loss_us', train_loss_us, epoch)
            writer.add_scalar('train/3.train_loss_r', train_loss_r, epoch)
            #writer.add_scalar('train/4.mask', mask_prob, epoch)
            writer.add_scalar('test/1.test_acc', test_acc, epoch)
            writer.add_scalar('test/2.test_loss', test_loss, epoch)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            #model_rot_to_save = model_rot.module if hasattr(model_rot, "module") else model_rot
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                #'state_dict2':model_rot_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                #'optimizer_rot':optimizer_rot.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

        test_accs.append(test_acc)
        logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('Mean top-1 acc: {:.2f}\n'.format(
            np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        writer.close()    

def random_rotate(x):
    b4 = x.shape[0] // 4
    x1 = x[:b4]
    x2 = torch.rot90(x[b4:], 1, [2,3])
    x3 = torch.rot90(x2[b4:], 1, [2,3])
    x4 = torch.rot90(x3[b4:], 1, [2,3])
    l = np.zeros(b4, np.int32)
    l = torch.from_numpy(np.concatenate([l, l + 1, l + 2, l + 3]))
    return torch.cat((x1,x2[:b4],x3[:b4],x4), dim=0), l

'''
def random_rotate(x):
    b4 = x.shape[0] // 4
    x, xt = x[:2 * b4], torch.transpose(x[2 * b4:], 3,2)
    l = np.zeros(b4, np.int32)
    l = torch.from_numpy(np.concatenate([l, l + 1, l + 2, l + 3]))
    return np.concatenate([x[:b4], torch.flip(x[b4:],[1]), torch.flip(xt[:b4], [0,1]), torch.flip(xt[b4:],[0])], axis=0), l
'''

def train(args, labeled_trainloader, unlabeled_trainloader,
         model, optimizer, ema_model, scheduler, epoch):
    if args.amp:
        from apex import amp
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_us = AverageMeter()
    losses_r = AverageMeter()
    losses_ua = AverageMeter()
    end = time.time()

    if not args.no_progress:
        p_bar = tqdm(range(args.iteration),
                     disable=args.local_rank not in [-1, 0])

    train_loader = zip(labeled_trainloader, unlabeled_trainloader)

    criterion = nn.CrossEntropyLoss()
    
    model.train()
    #model_rot.train()

    for batch_idx, (data_x, data_u) in enumerate(train_loader):                     #data_x : labeled data , data_u : unlabeled data
        inputs_x, targets_x = data_x                                                               
        (inputs_u_w, inputs_u_s), targets_uo = data_u                                       #inputs_u_w : unlabeled weak augmentation, inputs_u_s : unlabeled strong augmentation
        data_time.update(time.time() - end)
        batch_size = inputs_x.shape[0]

        # unlabeled inputs
        #inputs_u = torch.cat((inputs_u_w, inputs_u_s),dim=0)

        
        # rotation strong augmented unlabeled data
        rot_y, rot_l = random_rotate(inputs_u_s)
        rot_y = rot_y.to(args.device)
        logits_rot = model(rot_y)
        
        rot_l = rot_l.type(torch.int64)
        rot_l = rot_l.to(args.device)
        
        # supervised loss
        logits_x = model(inputs_x.to(args.device))
  
        # pseudo label
        with torch.no_grad():
        # compute guessed labels of unlabel samples 
        #need to fix for distribution alignment
            logits_u_temp = model(inputs_u_w.to(args.device))
            #q = torch.cat((torch.softmax(logits_u_w, dim=1), torch.softmax(logits_u_s, dim=1)), dim=0)
            q = torch.softmax(logits_u_temp,dim=1)
            q = q * (torch.softmax(logits_x, dim=1)).mean()/q.mean()
            
            pt = q**(1/0.5)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            pseudo_label = targets_u.detach()
            

        _, targets_u = torch.max(pseudo_label, dim=-1)
        targets_u = targets_u.long()
        
        # unlabeled strong loss
        logits_u_s = model(inputs_u_s.to(args.device))
        

        # concat & shuffle for mixup
        inputs_m = torch.cat((inputs_x,inputs_u_s),dim=0).to(args.device)
        #targets_m= torch.cat((targets_x.to(args.device), targets_u),dim=0).to(args.device)

        l = 0.75

        idx = torch.randperm(inputs_m.size(0))

        input_a, input_b = inputs_m, inputs_m[idx]
        #target_a, target_b = targets_m, targets_m[idx]
        mixed_input = l*input_a + (1-l)*input_b
        #mixed_target = l*target_a + (1-l)*target_b

        # interleave labeled and unlabeled samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input,batch_size))                     # mixed_input( 3 * batch_size) 를 batch_size개수만큼씩 자른다.
        mixed_input = interleave(mixed_input,batch_size)                            # mixed_output 도 마찬가지
        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))                                             # now logits is list.
                                                                                    # logits[0] : p_b
                                                                                    # logits[1:] : q_b
        # put interleaved samples back
        logits = interleave(logits,batch_size)
        logits_m_x = logits[0]
        logits_m_u = torch.cat(logits[1:],dim=0)

        #logits_m = model(mixed_input)

        #mixed_target = mixed_target.long()

        # unlabeld mixup loss
        #Lu = criterion(logits_m_u[len(logits_m_u)//2:], mixed_target[batch_size//2+len(mixed_target)//2:])

        # loss
        #Lx = criterion(logits_x,targets_x.to(args.device))
        
        # rotation losss
        loss_rot = F.cross_entropy(logits_rot, rot_l)
        #loss_rot=0
        Lx = criterion(logits_m_x, targets_x.to(args.device))
        #Lu = criterion(logits_m_u, mixed_target[batch_size:])
        Lu = criterion(logits_m_u, targets_u)
        Lus = (F.cross_entropy(logits_u_s, targets_u,reduction='none')).mean()
        
        # unlabeled guessed label answer check
        
        cnt=0
        for i,j in zip(targets_u, targets_uo):
            if i ==j:
                cnt += 1
        Lua = cnt / targets_u.size()[0]

        loss = Lx + 1.5*Lu + 0.5*Lus + 0.5*loss_rot
        
        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())
        losses_us.update(Lus.item())
        losses_ua.update(Lua)
        losses_r.update(loss_rot.item())

        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        scheduler.step()
        if args.use_ema:
            ema_model.update(model)

        model.zero_grad()
        #model_rot.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()
        #mask_prob = mask.mean().item()
        if not args.no_progress:
            p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Loss_us: {loss_us:.4f}. Loss_ua: {loss_ua:.4f}. Loss_r: {loss_r:.4f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                lr=scheduler.get_last_lr()[0],
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x=losses_x.avg,
                loss_u=losses_u.avg,
                loss_us=losses_us.avg,
                loss_ua=losses_ua.avg,
                loss_r=losses_r.avg
                #mask=mask_prob
                ))
            p_bar.update()
    if not args.no_progress:
        p_bar.close()
    return losses.avg, losses_x.avg, losses_u.avg, losses_us.avg, losses_r.avg


def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


class ModelEMA(object):
    def __init__(self, args, model, decay, device='', resume=''):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        self.wd = args.lr * args.wdecay
        if device:
            self.ema.to(device=device)
        self.ema_has_module = hasattr(self.ema, 'module')
        if resume:
            self._load_checkpoint(resume)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        assert isinstance(checkpoint, dict)
        if 'ema_state_dict' in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['ema_state_dict'].items():
                if self.ema_has_module:
                    name = 'module.' + k if not k.startswith('module') else k
                else:
                    name = k
                new_state_dict[name] = v
            self.ema.load_state_dict(new_state_dict)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema.state_dict().items():
                if needs_module:
                    k = 'module.' + k
                model_v = msd[k].detach()
                if self.device:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                # weight decay
                if 'bn' not in k:
                    msd[k] = msd[k] * (1. - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    cudnn.benchmark = True
    main()
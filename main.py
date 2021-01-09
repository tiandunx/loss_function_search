import torch
from torch import optim
from torch.utils.data import DataLoader
from backbone import create_backbone
from fc.fc_layers import FC
import os
from lmdb_utils import LMDBDataset, LMDBDistributeSampler
import argparse

import torch.nn.functional as F
from validate_lfw import *
import torch.distributed as dist
import logging as logger

logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
from lfs_core.loss_func_search import LossFuncSearch


class FaceNet(torch.nn.Module):
    def __init__(self, net_type, feat_dim, num_class):
        super(FaceNet, self).__init__()
        self.feat_net = create_backbone(net_type)
        #  As we perform loss search, it's recommended that we simply use normal full-connected layer.
        self.fc = FC('FC', embedding_size=feat_dim, num_class=num_class)

    def forward(self, x, label):
        feat = self.feat_net(x)
        logits = self.fc(feat, label)
        return logits


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_one_epoch(data_loader, model, optimizer, lfs, cur_epoch, device_id, print_freq, saved_freq, saved_dir):
    # switch to train mode
    model.train()
    db_size = len(data_loader)
    my_rank = dist.get_rank()
    start_iters = cur_epoch * db_size
    for batch_idx, (image, label) in enumerate(data_loader):
        image = image.cuda(device_id)
        label = label.cuda(device_id)
        pred = model(image, label)

        logits = F.softmax(pred, dim=1)
        loss = lfs.get_loss(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_iters = start_iters + batch_idx
        if cur_iters % print_freq == 0:
            loss_val = loss.item()
            lr = get_lr(optimizer)
            logger.info('rank %d, epoch %d, iter %d, lr %f, loss %f' % (my_rank, cur_epoch, batch_idx, lr, loss_val))

        if my_rank == 0 and cur_iters % saved_freq == 0:
            saved_name = 'epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            torch.save({'state_dict': model.module.feat_net.state_dict()}, os.path.join(saved_dir, saved_name))
            logger.info('save checkpoint %s to disk...' % saved_name)
    if my_rank == 0:
        saved_name = 'epoch_%d.pt' % cur_epoch
        torch.save({'state_dict': model.module.feat_net.state_dict()}, os.path.join(saved_dir, saved_name))
        logger.info('save checkpoint %s to disk...' % saved_name)


def train(conf):
    model = FaceNet(conf.net_type, conf.feat_dim, conf.num_class)

    start_device_id = conf.device * conf.num_gpus_per_rank
    end_device_id = (conf.device + 1) * conf.num_gpus_per_rank
    device_list = list(range(start_device_id, end_device_id))
    output_device = torch.device("cuda:%d" % device_list[0])
    model = torch.nn.DataParallel(model, device_list, device_list[0]).to(output_device)
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=1e-4)

    my_rank = dist.get_rank()
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=0.1)
    db = LMDBDataset(conf.source_lmdb_list, conf.source_file_list)
    lmdb_sampler = LMDBDistributeSampler(db, my_rank)
    data_loader = DataLoader(db, conf.batch_size, sampler=lmdb_sampler, num_workers=4)

    # create LFS
    lfs = LossFuncSearch(conf.scale, True if conf.do_search == 1 else False)
    lfs.set_model(model)

    for epoch in range(conf.epochs):
        lr_schedule.step()
        lfs.set_loss_parameters(epoch)
        train_one_epoch(data_loader, model, optimizer, lfs, epoch, device_list[0], conf.print_freq, conf.saved_freq, conf.saved_dir)
        lmdb_sampler.set_epoch(epoch)
        if epoch != 0 and (epoch + 1) % conf.val_freq == 0 and (epoch + 1) >= 3:
            reward = validate(model.feat_net, conf.test_pairs, conf.test_data_loader, output_device)
            logger.info('rank: %d, acc = %f' % (my_rank, reward))
            if conf.do_search == 1:
                lfs.update_lfs(reward)
    db.close()


def main(argv):
    conf = argparse.ArgumentParser(description='Loss Function Search for Face Recognition.')
    conf.add_argument('ip', type=str, )
    conf.add_argument('port', type=int)
    conf.add_argument('rank', type=int)
    conf.add_argument('world_size', type=int)
    conf.add_argument('source_lmdb_list', type=str, help='comma separated training dataset')
    conf.add_argument('source_file_list', type=str, help='comma separated training kv text file, echo line contains '
                                                         'lmdb_key label, it is space separated.')
    conf.add_argument("--saved_dir", type=str, default='snapshot', help='where to save the snapshot')
    conf.add_argument('--val_freq', type=int, default=1, help='when trained on margin-based.')
    conf.add_argument('--num_class', type=int, default=9809, help='number of categories')
    conf.add_argument('--net_type', type=str, default='mobile', choices=['mobile', 'r50', 'r101'])
    conf.add_argument('--feat_dim', type=int, default=512, help='feature dimension.')
    conf.add_argument('--scale', type=int, default=32, help='loss function search scale parameters.')
    conf.add_argument('--search_type', type=str, default='global', choices=['global', 'local'])
    conf.add_argument('--do_search', type=int, default=1, help='if 1, do loss search otherwise perform random softmax')
    conf.add_argument('--step', type=str, default='4,8,10',
                      help='similar to step specified in caffe solver, but in epoch mechanism')
    conf.add_argument('--epochs', type=int, default=20, help='how many epochs you want to train.')
    conf.add_argument('--print_freq', type=int, default=200, help='frequency of displaying current training state.')
    conf.add_argument('--saved_freq', type=int, default=5000, help='how often should we save the checkpoint.')
    conf.add_argument('--momentum', type=float, default=0.9, help='momentum')
    conf.add_argument('--batch_size', type=int, default=128, help='batch size over all gpus per rank.')

    conf.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'])
    conf.add_argument('--num_gpus_per_rank', type=int, default=4, help="#gpus each rank will use to train the model.")
    conf.add_argument('--lr', type=float, default=0.1, help='initial learning rate.')
    args = conf.parse_args()
    dist.init_process_group(backend=args.backend, init_method="tcp://%s:%d" % (args.ip, args.port),
                            world_size=args.world_size, rank=args.rank)
    args.milestones = [int(p) for p in args.step.split(',')]
    num_gpus_per_node = torch.cuda.device_count()
    if args.num_gpus_per_rank >= num_gpus_per_node:
        args.num_gpus_per_rank = max(1, num_gpus_per_node)

    test_pairs, test_data_loader = prepare_validate_data()
    args.test_pairs = test_pairs
    args.test_data_loader = test_data_loader
    args.num_gpus_per_node = num_gpus_per_node
    assert args.world_size * args.num_gpus_per_rank % args.num_gpus_per_node == 0
    node_id = args.rank * args.num_gpus_per_rank // num_gpus_per_node
    args.device = (args.rank * args.num_gpus_per_rank - node_id * args.num_gpus_per_node) // args.num_gpus_per_rank
    my_rank = dist.get_rank()
    if my_rank == 0:
        if not os.path.exists(args.saved_dir):
            os.makedirs(args.saved_dir)
    train(args)
    logger.info('Optimization done!')
    dist.destroy_process_group()


if __name__ == '__main__':
    main(sys.argv)

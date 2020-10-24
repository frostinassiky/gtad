import os
import torch
import torch.nn.parallel
import torch.optim as optim
from torch import autograd
import numpy as np

from gtad_lib import opts
from gtad_lib.models import GTAD
from gtad_lib.dataset import VideoDataSet
from gtad_lib.loss_function import get_mask, gtad_loss_func # subgraph_loss_func, node_loss_func


################## fix everything ##################
import random
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#######################################################

# keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.sum += val
        self.count += n
    def avg(self):
        return self.sum/self.count

def get_mem_usage():
    GB = 1024.0 ** 3
    output = ["device_%d = %.03fGB" % (device, torch.cuda.max_memory_allocated(torch.device('cuda:%d' % device)) / GB) for device in range(opt['n_gpu'])]
    return ' '.join(output)[:-1]


# train
def train(data_loader, model, optimizer, epoch, bm_mask):
    model.train()
    total_am, subgraph_am, node_am = AverageMeter(), AverageMeter(), AverageMeter()
    for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):
        # forward pass
        confidence_map, start, end = model(input_data.cuda())
        # loss
        loss = gtad_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        # update step
        optimizer.zero_grad()
        loss[0].backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # update losses
        total_am.update(loss[0].detach())
        subgraph_am.update(loss[1].detach())
        node_am.update(loss[2].detach())

    print("[Epoch {0:03d}]\tLoss {1:.2f} = {2:.2f} + {3:.2f} (train)".format(
        epoch, total_am.avg(), subgraph_am.avg(), node_am.avg()))


def test(data_loader, model, epoch, bm_mask, best_loss):
    model.eval()
    total_am, subgraph_am, node_am = AverageMeter(), AverageMeter(), AverageMeter()
    with torch.no_grad():
      for n_iter, (input_data, label_confidence, label_start, label_end) in enumerate(data_loader):

        # forward pass
        confidence_map, start, end = model(input_data.cuda())
        # loss
        # gt_iou_map = label_confidence.cuda() * bm_mask
        loss = gtad_loss_func(confidence_map, start, end, label_confidence, label_start, label_end, bm_mask.cuda())

        # update losses
        total_am.update(loss[0].detach())
        subgraph_am.update(loss[1].detach())
        node_am.update(loss[2].detach())

    print("[Epoch {0:03d}]\tLoss {1:.2f} = {2:.2f} + {3:.2f} (validation)".format(
        epoch, total_am.avg(), subgraph_am.avg(), node_am.avg()))

    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    torch.save(state, opt["output"] + "/GTAD_checkpoint.pth.tar")
    if total_am.avg() < best_loss:
        best_loss = total_am.avg()
        torch.save(state, opt["output"] + "/GTAD_best.pth.tar")

    return best_loss

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists(opt["output"]):
        os.makedirs(opt["output"])

    # model = GTAD(opt)
    # a = torch.randn(1, 400, 100)
    # b, c = model(a)
    # print(b.shape, c.shape)
    # print(b)
    # print(c)

    model = GTAD(opt)

    model = torch.nn.DataParallel(model, device_ids=list(range(opt['n_gpu']))).cuda()
    print('use {} gpus to train!'.format(opt['n_gpu']))

    optimizer = optim.Adam(model.parameters(), lr=opt["training_lr"],
                           weight_decay=opt["weight_decay"])
    train_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="train"),
                                               batch_size=opt["batch_size"], shuffle=True,
                                               num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=opt["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt["step_size"], gamma=opt["step_gamma"])
    mask = get_mask(opt["temporal_scale"])
    best_loss = 1e10
    for epoch in range(opt["train_epochs"]):
      with autograd.detect_anomaly():
        train(train_loader, model, optimizer, epoch, mask)
        best_loss = test(test_loader, model, epoch, mask, best_loss)
        scheduler.step()




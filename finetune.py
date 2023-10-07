import argparse
import os
import shutil
import time

import mlflow
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader 
from tqdm import tqdm
import torch.nn.functional as F

from datasets import ImageDataset2
from models import UNet
from utils import LambdaLR, dice_loss, split_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./datasets/FS-CAD', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--fold_id', type=int, default=0, help='test fold id')
parser.add_argument('--fold_n', type=int, default=5, help='number of fold')
parser.add_argument('--wd', type=float, default=0., help='weight decay')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--nd', type=int, default=1, help='train the discriminator every nd steps')
parser.add_argument('--ng', type=int, default=1, help='train the generator every ng steps')
parser.add_argument('--dim', type=int, default=32, help='network base dim')
parser.add_argument('--device', type=str, default='cpu', help='select device, such as cpu,cuda:0,mgpu use cuda')
parser.add_argument('--log_step', type=int, default=100, help='log to mlflow every step')
parser.add_argument('--sleep', type=float, default=0., help='slow down train, if you gpu overheat, general 0.2')
parser.add_argument('--exp_name', type=str, help='mlflow experiment name', default='finetune_fscad')
parser.add_argument('--name', type=str, help='mlflow trial name', required=True)
parser.add_argument('--ckpt', type=str, default=None, help='load from checkpoint')
parser.add_argument('--nobi', action='store_false', help='unet bilinear mode')
parser.add_argument('--seed', type=int, default=42, help='random seed')

opt = parser.parse_args()
print(opt)

device = torch.device(opt.device)
###### Definition of variables ######
# Networks
netE = UNet(opt.output_nc, opt.input_nc, dim=opt.dim, bilinear=opt.nobi, res=False)  # U

if opt.device != 'cpu':
    netE.to(device)
    if opt.device == 'cuda':
        netE = DataParallel(netE.to(device))

# Set logger
mlflow.set_experiment(opt.exp_name)
run = mlflow.start_run(run_name=opt.name)
run_id = run.info.run_id
print('run id: ', run_id)
experiment_id = run.info.experiment_id
run_dir = f'mlruns/{experiment_id}/{run_id}'
art_dir = f"{run_dir}/artifacts"
ckpt_path = f"{run_dir}/last.ckpt"
mlflow.log_params(vars(opt))
source_code = [i for i in os.listdir() if ".py" in i]
for i in source_code:
    shutil.copy(i, f"{art_dir}/{i}")

# Load from checkpoint
if opt.ckpt is not None:
    checkpoint = torch.load(opt.ckpt)
    netE.load_state_dict(checkpoint['netE'])
    print(f'load from checkpoint: {opt.ckpt}')
    # forzen layer fine tune
    #  for param in netE.parameters():
        #  param.requires_grad = False
    #  for param in netE.module.outc.parameters():
        #  param.requires_grad = True
    #  for param in netE.module.up4.parameters():
        #  param.requires_grad = True

optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, netE.parameters()), lr=opt.lr, weight_decay=opt.wd, foreach=True)
#  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, netE.parameters()), lr=opt.lr, weight_decay=opt.wd)
lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
input_A = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size, opt.size).to(device)
input_B = torch.FloatTensor(opt.batch_size, opt.output_nc, opt.size, opt.size).to(device)

# fold
ds = ImageDataset2(opt.dataroot, size=opt.size, unaligned=False, mode='test',return_img_name=False)
#  train_ratio = 4/5
#  train_size = int(len(ds) * train_ratio)
#  valid_size = len(ds) - train_size
#  train_ds, valid_ds = random_split(ds, [train_size, valid_size], generator=torch.Generator().manual_seed(opt.seed))


train_ds, valid_ds = split_dataset(ds, opt.fold_n, opt.fold_id)

train_dl = DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=opt.n_cpu)


###### Training ######
loss_fn = torch.nn.BCEWithLogitsLoss()
step = 0
best_dice_score = 0
for epoch in range(opt.epoch, opt.n_epochs):
    pbar = tqdm(train_dl)
    for i, batch in enumerate(pbar):
        step += 1
        # Set model input
        x = batch['A'].to(device)
        y = batch['B'].to(device)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)


        pred_y = netE(x)
        #  pred_y = torch.sigmoid(pred_y)
        y = (y + 1) / 2.0
        loss1 = loss_fn(pred_y, y)
        loss2 = dice_loss(F.sigmoid(pred_y), y)
        loss = loss1 + loss2

        loss.backward()

        optimizer.step()

        # Display results
        pbar.set_description(f'Epoch:{epoch}')
        pbar.set_postfix_str(f'loss_bce={loss1.item():.4}, loss_dice={loss2.item():.4} lr={optimizer.param_groups[0]["lr"]}')
        if i % opt.log_step == 0:
            mlflow.log_metrics({'loss': loss.item()}, step=step)
        time.sleep(opt.sleep)

    # Update learning rates
    lr_scheduler.step()
    

    # eval
    pbar_eval = tqdm(valid_dl)
    dice_score_list = []
    for i, batch in enumerate(pbar_eval):
        x = batch['A'].to(device).unsqueeze(1)
        y = batch['B'].to(device).unsqueeze(1)
        pred_y = netE(x)
        pred_y = torch.sign(pred_y)
        dice_score = 1 - dice_loss((pred_y+1)/2, (y+1)/2)
        dice_score_list.append(dice_score.item())

    dice_score = sum(dice_score_list) / len(dice_score_list)
    mlflow.log_metric('valid_dice_score', dice_score, step=epoch)
    pbar.write(f'Epoch:{epoch} dice_score:{dice_score:.4}')

    # save best model
    if dice_score > best_dice_score:
        best_dice_score = dice_score
        states = {'netE': netE.state_dict(),
                  'current_epoch': epoch}
        torch.save(states, f'{run_dir}/best.ckpt')

    # Save last checkpoints
    states = {'netE': netE.state_dict(),
              'optimizer': optimizer.state_dict(),
              'current_epoch': epoch}
    torch.save(states, ckpt_path)


mlflow.end_run()

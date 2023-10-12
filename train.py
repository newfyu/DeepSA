import argparse
import itertools
import os
import shutil
import time

import mlflow
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import numpy as np

from datasets import ImageDataset
from models import Discriminator, Generator, UNet, MultiscaleDiscriminator
from utils import EMA, LambdaLR, ReplayBuffer, weights_init_normal,SegmentationMetric,out2mask, batch2pil

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=60, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=40, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/LM-CAD/', help='root directory of the dataset')
parser.add_argument('--lr_G', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--lr_D', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--nd', type=int, default=1, help='train the discriminator every nd steps')
parser.add_argument('--ng', type=int, default=1, help='train the generator every ng steps')
parser.add_argument('--dim', type=int, default=32, help='network base dim')
parser.add_argument('--w_idt', type=int, default=5, help='idt loss weight')
parser.add_argument('--w_cycle', type=int, default=10, help='cycle loss weight')
parser.add_argument('--w_a2b', type=int, default=1, help='GAN generator_A2B loss weight')
parser.add_argument('--w_b2a', type=int, default=1, help='GAN generator_B2A loss weight')
#  parser.add_argument('--w_mrr', type=float, default=0, help='Minimum residual regular loss weight')
parser.add_argument('--replay_prob', type=float, default=1, help='replay prob')
parser.add_argument('--device', type=str, default='cpu', help='select device, such as cpu,cuda:0,mgpu use cuda')
parser.add_argument('--log_step', type=int, default=100, help='log to mlflow every step')
parser.add_argument('--ema_step', type=int, default=100, help='ema update step, default 10')
parser.add_argument('--ema_begin_step', type=int, default=0, help='ema start step')
parser.add_argument('--sleep', type=float, default=0., help='slow down train, if you gpu overheat, general 0.2')
parser.add_argument('--exp_name', type=str, help='mlflow experiment name', default='V46Eval')
parser.add_argument('--name', type=str, help='mlflow trial name', required=True)
parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint, also need set n_epochs,decay_epoch')
parser.add_argument('--nobi', action='store_false', help='unet bilinear mode')

opt = parser.parse_args()
print(opt)

device = torch.device(opt.device)
###### Definition of variables ######
# Networks
#  netG_A2B = Generator(opt.input_nc, opt.output_nc) # G
#  netG_B2A = Generator(opt.output_nc, opt.input_nc) # G
netG_A2B = UNet(opt.input_nc, opt.output_nc, dim=opt.dim, bilinear=opt.nobi)  # U
netG_B2A = UNet(opt.output_nc, opt.input_nc, dim=opt.dim, bilinear=opt.nobi)  # U
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

# EMA model
ema_updater = EMA(0.995)
#  netE = Generator(opt.output_nc, opt.input_nc) # G
netE = UNet(opt.output_nc, opt.input_nc, dim=opt.dim, bilinear=opt.nobi)  # U

if opt.device != 'cpu':
    netG_A2B.to(device)
    netG_B2A.to(device)
    netD_A.to(device)
    netD_B.to(device)
    netE.to(device)
    if opt.device == 'cuda':
        netG_A2B = DataParallel(netG_A2B.to(device))
        netG_B2A = DataParallel(netG_B2A.to(device))
        netD_A = DataParallel(netD_A.to(device))
        netD_B = DataParallel(netD_B.to(device))
        netE = DataParallel(netE.to(device))

#  netG_A2B.apply(weights_init_normal) # G
#  netG_B2A.apply(weights_init_normal) # G
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=opt.lr_G, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))

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
if opt.resume is not None:
    checkpoint = torch.load(opt.resume)
    opt.epoch = checkpoint['current_epoch'] + 1
    netG_A2B.load_state_dict(checkpoint['netG_A2B'])
    netG_B2A.load_state_dict(checkpoint['netG_B2A'])
    netD_A.load_state_dict(checkpoint['netD_A'])
    netD_B.load_state_dict(checkpoint['netD_B'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
    optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
    print(f'find ckpt, load from checkpoint: {opt.resume}, epoch is {opt.epoch}')

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
input_A = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size, opt.size).to(device)
input_B = torch.FloatTensor(opt.batch_size, opt.output_nc, opt.size, opt.size).to(device)
input_eval_B = torch.FloatTensor(1, opt.output_nc, opt.size, opt.size).to(device)
target_real = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(1.0), requires_grad=False).to(device)
target_fake = Variable(torch.FloatTensor(opt.batch_size, 1).fill_(0.0), requires_grad=False).to(device)

fake_A_buffer = ReplayBuffer(p=opt.replay_prob)
fake_B_buffer = ReplayBuffer(p=opt.replay_prob)

# Dataset loader
dataloader = DataLoader(ImageDataset(opt.dataroot, unaligned=True, size=opt.size, mode='train'),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

simple_dl = DataLoader(ImageDataset(opt.dataroot, unaligned=False, size=opt.size, mode='test'),
                       batch_size=8, shuffle=False, num_workers=opt.n_cpu)

eval_dl = DataLoader(ImageDataset('./', size=opt.size, unaligned=False, mode='eval50',return_img_name=True),
                     batch_size=1, shuffle=False, num_workers=opt.n_cpu)

fix_sample = next(iter(simple_dl))
fix_A = fix_sample['A'].to(device)
fix_B = fix_sample['B'].to(device)

metric = SegmentationMetric(2)
###################################

###### Training ######
step = 0
for epoch in range(opt.epoch, opt.n_epochs):
    pbar = tqdm(dataloader)
    for i, batch in enumerate(pbar):
        step += 1
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss
        #  G_A2B(B) should equal B if real B is fed
        same_B = netG_A2B(real_B)
        loss_identity_B = criterion_identity(same_B, real_B) * opt.w_idt
        # G_B2A(A) should equal A if real A is fed
        same_A = netG_B2A(real_A)
        loss_identity_A = criterion_identity(same_A, real_A) * opt.w_idt

        # GAN loss
        fake_B = netG_A2B(real_A)
        pred_fake = netD_B(fake_B)
        loss_GAN_A2B = criterion_GAN(pred_fake, target_real) * opt.w_a2b

        fake_A = netG_B2A(real_B)
        pred_fake = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(pred_fake, target_real) * opt.w_b2a
        #  loss_mrr = criterion_identity(fake_A, real_B) * opt.w_mrr

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * opt.w_cycle

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * opt.w_cycle

        # Total loss
        #  loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_mrr
        loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
        loss_G.backward()

        if step % opt.ng == 0:
            optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5
        loss_D_A.backward()

        if step % opt.nd == 0:
            optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######
        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = netD_B(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = netD_B(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5
        loss_D_B.backward()

        if step % opt.nd == 0:
            optimizer_D_B.step()
        ###################################
        # ema update
        if step != 0 and step % opt.ema_step == 0 and step > opt.ema_begin_step:
            ema_updater.update_moving_average(netE, netG_B2A)

        pbar.set_description(f'Epoch:{epoch}')
        pbar.set_postfix_str(f'loss={loss_G:.4}, idt={loss_identity_A + loss_identity_B:.4}, G={loss_GAN_A2B + loss_GAN_B2A:.4}, cycle={loss_cycle_ABA + loss_cycle_BAB:.4}, D={loss_D_A + loss_D_B:.4}, lr={optimizer_G.param_groups[0]["lr"]}')
        if i % opt.log_step == 0:
            mlflow.log_metrics({'loss_G': loss_G.item(), 'loss_G_identity': (loss_identity_A + loss_identity_B).item(), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A).item(), 'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB).item(), 'loss_D': (loss_D_A + loss_D_B).item()}, step=step)
        time.sleep(opt.sleep)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    # Image sample
    with torch.no_grad():
        netG_B2A.eval()
        if opt.device == 'cuda':
            out = netG_B2A.module.model(fix_B)
        else:
            out = netG_B2A.model(fix_B)
        fake_A = out + fix_B
        out2 = (out > 0.01).float()
        # Gen ema image
        if opt.device == 'cuda':
            out_ema = netE.module.model(fix_B)
        else:
            out_ema = netE.model(fix_B)
        fake_E = out_ema + fix_B
        out_ema2 = (out_ema > 0.01).float()

        B_norm = make_grid(fix_B, normalize=True, padding=0)
        fake_A = make_grid(fake_A, normalize=True, padding=0)
        out = make_grid(out, normalize=True, padding=0)
        out2 = make_grid(out2, normalize=True, padding=0)
        out_ema = make_grid(out_ema, normalize=True, padding=0)
        out_ema2 = make_grid(out_ema2, normalize=True, padding=0)

        imgs = make_grid([B_norm, fake_A, out, out2, out_ema, out_ema2], normalize=True, nrow=1)
        save_image(imgs, f'{art_dir}/img_{str(epoch).zfill(4)}.png')
        netG_B2A.train()

    # eval
    metric.reset()
    pbar_eval = tqdm(eval_dl)
    preds = []
    for i, batch in enumerate(pbar_eval):
        real_B = Variable(input_eval_B.copy_(batch['B']))
        B_name = batch['B_name'][0]
        if opt.device == 'cuda':
            out = netE.module.model(real_B)
        else:
            out = netE.model(real_B)
        pred = out2mask(out)
        pred= np.array(pred.convert('1')).astype(int)
        gt = Image.open(f'eval50/GT/{B_name}.png')
        gt = np.array(gt).astype(int)
        metric.addBatch(pred,gt)
        preds.append(pred)

    eval_iou = metric.IntersectionOverUnion()[1]
    mlflow.log_metric('eval_iou',eval_iou,step=epoch)
    eval_imgs = batch2pil([torch.from_numpy(x).float().unsqueeze(0) for x in preds],nrow=5)
    eval_imgs.save(f'{art_dir}/eval50_{str(epoch).zfill(4)}.png')


    # Save last checkpoints
    states = {'netG_A2B': netG_A2B.state_dict(),
              'netG_B2A': netG_B2A.state_dict(),
              'netD_A': netD_A.state_dict(),
              'netD_B': netD_B.state_dict(),
              'netE': netE.state_dict(),
              'optimizer_G': optimizer_G.state_dict(),
              'optimizer_D_A': optimizer_D_A.state_dict(),
              'optimizer_D_B': optimizer_D_B.state_dict(),
              'current_epoch': epoch}
    torch.save(states, ckpt_path)

    # Save every epoch for B2A
    states = {
        'netG_B2A': netG_B2A.state_dict(),
        'netD_A': netD_A.state_dict(),
        'netE': netE.state_dict()
    }
    torch.save(states, f'{run_dir}/{str(epoch).zfill(3)}.ckpt')

mlflow.end_run()

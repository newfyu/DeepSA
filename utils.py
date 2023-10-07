import random

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageOps
from pydicom import dcmread
from torch.autograd import Variable
from tqdm import tqdm
from skimage import filters, morphology
from datasets import tophat
from torch.utils.data import ConcatDataset

def batch2pil(x, nrow=8, normalize=True, padding=1, pad_value=1, range=None):
    grid = torchvision.utils.make_grid(
        x, normalize=normalize, nrow=nrow, pad_value=pad_value, padding=padding, range=range
    )
    return torchvision.transforms.ToPILImage()(grid.cpu())

def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class ReplayBuffer():
    def __init__(self, max_size=50, p=0.5):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []
        self.p = p

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > self.p:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.started = False

    def update_average(self, old, new):
        if not self.started:
            self.started = True
            return new
        return old * self.beta + (1 - self.beta) * new

    def update_moving_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)


def fusion_predict(model, ckpts, x, size=512, pad=0, device='cpu', return_x=True, multiangle=False, denoise=3, cutoff=1, padding_mode='reflect', netE=True):
    """融合多个角度或多个ckpt的输出,可取得更好的视觉效果
    ckpt：checkpoint path
    x: input multiple tensor, each shape(C,H,W)
    pad：边缘填充, 如果mutiangle=False, 仅填充right和bottom，如果multiangel=True, 填充四边。
         pad可增加边缘血管的提取，但也可能增加噪声
    return_x: 是否返回转换成图片的x
    multiangle: 是否多角度预测
    denoise: 去噪强度
    """
    outs = []
    B0 = x.to(device)
    if multiangle:
#         B1 = T.functional.rotate(B0, 90)
#         B2 = T.functional.rotate(B0, 180)
#         B3 = T.functional.rotate(B0, 270)
#         B = torch.stack((B0, B1, B2, B3))
        
        B1 = T.functional.hflip(B0)
        B2 = T.functional.vflip(B0)
        B = torch.stack((B0, B1, B2))
        B = T.functional.pad(B, pad, padding_mode=padding_mode)
    else:
        B = B0.unsqueeze(0)
        B = T.functional.pad(B, (0, 0, pad, pad), padding_mode=padding_mode)  # 仅pad了底边
    if return_x:
        B_dnorm = torchvision.utils.make_grid(B0, normalize=True, padding=0)
        B_dnorm = T.ToPILImage()(B_dnorm)
        B_dnorm = T.CenterCrop(size)(B_dnorm)

    for ckpt in ckpts:
        if isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            try:
                checkpoint = torch.load(ckpt, map_location=device)
                if netE:
                    model.load_state_dict(checkpoint['netE'])
                else:
                    model.load_state_dict(checkpoint['netG_B2A'])
            except:
                pass

        model.to(device)
        with torch.no_grad():
            try:
                fakeA = model.model(B)
            except:
                fakeA = model.module.model(B)
        if multiangle:
#             fakeA[1] = T.functional.rotate(fakeA[1], 270)
#             fakeA[2] = T.functional.rotate(fakeA[2], 180)
#             fakeA[3] = T.functional.rotate(fakeA[3], 90)

            fakeA[1] = T.functional.hflip(fakeA[1])
            fakeA[2] = T.functional.vflip(fakeA[2])
            fakeA = T.CenterCrop(size)(fakeA)
        else:
            fakeA = fakeA[:, :, :size, :size]

        outs.append(fakeA)

    out = torch.cat(outs)
    out = out.max(0, True)[0]
#     out = out.mean(0, True)[0]
    out = torchvision.utils.make_grid(out, normalize=True)

    out = (out.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
    out = cv2.fastNlMeansDenoising(out, None, denoise, 7, 21)

    out = T.ToPILImage()(out)
    out = ImageOps.autocontrast(out, cutoff=cutoff)

    if return_x:
        return B_dnorm, out
    else:
        return out
    
def out2mask(out):
    out = torchvision.utils.make_grid(out, normalize=True)
    out = (out.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
    out = cv2.fastNlMeansDenoising(out, None, 3, 7, 21)
    out = T.ToPILImage()(out)
    out = ImageOps.autocontrast(out, cutoff=0.5)
    out = make_mask(out,)
    return out


def make_gif_from_dicom(src, dst, model, ckpts, pad=0, device='cpu', multiangle=True, denoise=5, cutoff=1, gamma=1.5, size=512):
    """读取dicom，提取血管后转换为gif图片
    scr: dicom地址
    dst: 输出gif地址
    model: 输入模型nn
    ckpts: 模型的checkpoint，list，可以多个
    pad：边缘填充, 如果mutiangle=False, 仅填充right和bottom，如果multiangel=True, 填充四边
    device: 设备号，比如'cpu','cuda:0'
    输出: gif
    """
    arr = dcmread(src).pixel_array
    tfmc = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    tfmc2 = T.Compose([T.Resize(size), T.ToTensor()])
    imgs = []
    for i in tqdm(range(arr.shape[0])):
        B_arr = arr[i]
        B = tophat(B_arr)
        B = tfmc(B)
        B_dnorm, fakeA = fusion_predict(model, ckpts, B, pad=pad, device=device, multiangle=multiangle, denoise=denoise, cutoff=cutoff, size=size)
        B_dnorm = tfmc2(Image.fromarray(B_arr)).repeat(3, 1, 1)
        B_dnorm = torchvision.utils.make_grid(B_dnorm, normalize=True, padding=0)
        fakeA = T.functional.adjust_gamma(fakeA, gamma=gamma)
        fakeA = T.ToTensor()(fakeA)
        fakeA = 1 - fakeA
        
        # 降低曝光度，减少白色刺眼背景
        fakeA = fakeA * (2**(-0.5))
        
        grid = torch.stack((B_dnorm, fakeA), dim=0)
        grid = torchvision.utils.make_grid(grid)
        img = T.ToPILImage()(grid)
        

        imgs.append(img)
    img.save(dst, save_all=True, append_images=imgs)


def make_mask(img, local_kernel=21, local_offset=0, yan_offset=0, close_iter=3, remove_size=2000, hole_max_size=50):
    """
    local_kernel: thresh_local's filter size
    local_offset: thresh of local's offset value
    yan_offset: thresh of yan's offset value
    close_iter: image close operate's iter number
    remove_size: remove_small_objects's max size
    """
    image = np.array(img.convert('L'))
    thresh = filters.threshold_yen(image)
    seg1 = (image >= (thresh - yan_offset))
    seg1 = morphology.remove_small_objects(seg1, 30)
    seg1 = seg1.astype('uint8') * 255

    thresh_local = filters.threshold_local(image, local_kernel)
    seg2 = (image >= (thresh_local - local_offset))
    seg2 = morphology.remove_small_objects(seg2, 30)
    seg2 = seg2.astype('uint8') * 255

    inter = ((seg2 / 255) * (seg1 / 255) * 255).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dst2 = cv2.morphologyEx(inter, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    dst2 = morphology.remove_small_objects((dst2 / 255).astype('bool'), remove_size)
    dst2 = (dst2 * 255).astype('uint8')

    inter = ((dst2 / 255) * (inter / 255) * 255).astype('uint8')
    inter = fill_hole(inter, hole_max_size=hole_max_size)
    return T.ToPILImage()(inter)
    
    #  if return_skel:
        #  skel = morphology.skeletonize(inter / 255)
        #  skel = skel.astype(np.uint8) * 255
        #  dst_rgb = cv2.cvtColor(inter.astype('uint8'), cv2.COLOR_GRAY2RGB)
        #  dst_rgb[:, :, 1] += (skel / 50).astype('uint8')
        #  #  dst_rgb[:, :, 2] += (skel / 50).astype('uint8')
        #  return T.ToPILImage()(inter), T.ToPILImage()(dst_rgb)
    #  else:
        #  contours2,_ = cv2.findContours(inter, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #  edge = np.zeros_like(img).astype('uint8')
        #  for i in range(len(contours2)):
            #  cv2.drawContours(edge, contours2, i, (255,255,255))
        #  return T.ToPILImage()(inter), T.ToPILImage()(edge)


def merge_ckpts(ckpt_list):
    """
    融合多个ckpt
    """
    ckpts = ckpt_list
    ckpt = ckpts[0].copy()
    for key in ckpt.keys():
        ckpt[key] = sum([c[key] for c in ckpts]) / len(ckpts)
    return ckpt


def denorm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-5)

def fill_hole(img, hole_max_size=50):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)  # 轮廓的个数
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= hole_max_size:
            cv_contours.append(contour)
        else:
            continue
    cv2.fillPoly(img, cv_contours, (255, 255, 255))
    return img

def clear_mask(mask, area_max=3000):
    sp = np.array(mask)
    conts, _ = cv2.findContours(sp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for idx, cont in enumerate(conts):
        area = cv2.contourArea(cont)
        perimeter= cv2.arcLength(cont,True)
        apr = area/(perimeter+1e-5)
        if apr>2.5 and area<area_max:
            cv2.fillPoly(sp, [cont], (0,0,0))
    return Image.fromarray(sp)


class SegmentationMetric(object):
    """分割函数的指标"""

    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU
    
    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass) # mask>=0
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
            np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
            np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def split_dataset(dataset, fold_n, fold_id):
    assert 0 <= fold_id < fold_n, "fold_id should be in the range [0, fold_n-1]"
    total_size = len(dataset)
    fold_size = total_size // fold_n
    sizes = [fold_size] * fold_n
    # 为了处理不整除的情况，将多余的样本加到最后一个fold
    sizes[-1] += total_size - sum(sizes)
    
    folds = []
    start_idx = 0
    for s in sizes:
        end_idx = start_idx + s
        folds.append(torch.utils.data.Subset(dataset, list(range(start_idx, end_idx))))
        start_idx = end_idx
    
    valid_ds = folds[fold_id]
    train_ds = ConcatDataset(folds[:fold_id] + folds[fold_id+1:])
    
    return train_ds, valid_ds

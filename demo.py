import gradio as gr
import torch
from models import UNet
from datasets import tophat
import torchvision.transforms as T
import numpy as np
from PIL import Image,ImageChops
from utils import fusion_predict, make_mask, clear_mask
from pathlib import Path

SIZE = 512

tfmc1 = T.Compose([
    T.Resize(SIZE),
    T.Lambda(lambda img: tophat(img,50)),
    T.ToTensor(),
    T.Normalize((0.5),(0.5))
])

tfmc2 = T.Compose([
    T.Resize(SIZE),
    T.ToTensor(),
    T.Normalize((0.5),(0.5))
])

netG_B2A = UNet(1, 1, 32, bilinear=True)
netG_B2A = torch.nn.DataParallel(netG_B2A)

ckpts = ['ckpt/fscad_36249.ckpt']

def predict(img, options):
    if "Multiangle" in options:
        multiangle = True
    else:
        multiangle = False
    if "Pad margin" in options:
        pad = 50
    else:
        pad = 0

    img =  img.convert('L')
    x1 = tfmc1(img)
    x2 = tfmc2(img)
    _, out1 = fusion_predict(netG_B2A, ckpts, x1, multiangle=multiangle, denoise=4, size=SIZE, cutoff=0.4, pad=pad, netE=True)
    _, out2 = fusion_predict(netG_B2A, ckpts, x2, multiangle=False, denoise=4, size=SIZE, cutoff=0.4, pad=pad, netE=True)

    out_merge = Image.fromarray(np.expand_dims(np.max(np.concatenate((np.array(out1),np.array(out2)),axis=2),axis=2),2).repeat(3,2))

    mask_merge = make_mask(out_merge,remove_size=2000, local_kernel=21, hole_max_size=100)
    out_merge = T.functional.adjust_gamma(out_merge, 2)
    mask_cld = clear_mask(mask_merge)

    out_ts = ImageChops.invert(out_merge)
    out_ts = T.ToTensor()(out_ts)
    out_ts = out_ts * (2**(-0.5))
    out_ts = T.ToPILImage()(out_ts)

    return out_ts, mask_cld


title = "DeepSA"
description = "Deep Subtraction Angiography"
article = "<p style='text-align: center'><a href='https://github.com/bryandlee/animegan2-pytorch' target='_blank'>Github Repo Pytorch</a></p> <center><img src='https://visitor-badge.glitch.me/badge?page_id=akhaliq_animegan' alt='visitor badge'></center></p>"

# 读取example文件夹下的png图片的路径
examples = Path("example").glob("*.png")
examples = [[str(e)] for e in examples]

demo = gr.Interface(
    fn=predict, 
    inputs=[gr.inputs.Image(type="pil"), gr.inputs.CheckboxGroup(["Multiangle","Pad margin"], label="Options")],
    outputs=[gr.outputs.Image(type="pil",label='Deep subtraction'), gr.outputs.Image(type="pil", label='Vessel segmentation')],
    title=title,
    description=description,
    article=article,
    examples=examples
)

demo.launch(server_name='0.0.0.0')


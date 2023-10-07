import gradio as gr
import torch
from models import UNet
from datasets import tophat
import torchvision.transforms as T
from torchvision.utils import make_grid
import numpy as np
from PIL import Image, ImageChops, ImageOps
from utils import fusion_predict, make_mask, clear_mask
from pathlib import Path
import cv2
from skimage import morphology

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


ckpts = ['ckpt/fscad_36249.ckpt'] # 828 FS-Model
#  ckpts = ['ckpt/xcad_4afe3.ckpt'] # XCAD-Model
#  ckpts = ['ckpt/pt_bc62a.ckpt'] # PT-Model

netE = UNet(1, 1, 32, bilinear=True)
checkpoint = torch.load(ckpts[0], map_location="cpu")
new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['netE'].items()}
netE.load_state_dict(new_state_dict)
netE.to('cpu')

def predict(img, auto_tresh, options):
    if auto_tresh:
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
        _, out1 = fusion_predict(netE, ["none"], x1, multiangle=multiangle, denoise=4, size=SIZE, cutoff=0.4, pad=pad, netE=True)
        _, out2 = fusion_predict(netE, ["none"], x2, multiangle=False, denoise=4, size=SIZE, cutoff=0.4, pad=pad, netE=True)

        out_merge = Image.fromarray(np.expand_dims(np.max(np.concatenate((np.array(out1),np.array(out2)),axis=2),axis=2),2).repeat(3,2))

        mask_merge = make_mask(out_merge,remove_size=2000, local_kernel=21, hole_max_size=100)
        out_merge = T.functional.adjust_gamma(out_merge, 2)
        seg_img = clear_mask(mask_merge)

        sub_img = ImageChops.invert(out_merge)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2**(-0.5))
        sub_img = T.ToPILImage()(sub_img)
    else:
        img =  img.convert('L')
        x = tfmc1(img)
        input = x.unsqueeze(0)
        with torch.no_grad():
            pred_y = netE(input)
        
        # 处理减影图片
        sub_img = make_grid(pred_y,normalize=True)
        sub_img = (sub_img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype('uint8')
        sub_img = cv2.fastNlMeansDenoising(sub_img, None, 3, 7, 21)

        sub_img = T.ToPILImage()(sub_img)
        sub_img = ImageOps.autocontrast(sub_img, cutoff=1)
        sub_img = T.functional.adjust_gamma(sub_img, 2)

        sub_img = ImageChops.invert(sub_img)
        sub_img = T.ToTensor()(sub_img)
        sub_img = sub_img * (2**(-0.5))
        sub_img = T.ToPILImage()(sub_img)
        
        
        # 处理分割图片
        seg_img = torch.sign(pred_y)
        seg_img = ((seg_img.cpu().detach() + 1)/2).numpy().astype(bool)
        seg_img = morphology.remove_small_objects(seg_img, 500)
        seg_img = (seg_img * 255).astype('uint8')
        seg_img = torch.from_numpy(seg_img/255)
        seg_img = T.ToPILImage()(seg_img[0])

    return sub_img, seg_img



title = "DeepSA"
description = "Deep Subtraction Angiography"
article = "<p style='text-align: center'><a href='https://github.com/newfyu/DeepSA' target='_blank'>Github Repo</a>"

examples = Path("example").glob("*.png")
examples = [[str(e)] for e in examples]

demo = gr.Interface(
    fn=predict, 
    inputs=[gr.inputs.Image(type="pil"), gr.Checkbox(value=True, label="AutoTresh"), gr.inputs.CheckboxGroup(["Multiangle","Pad margin"], label="Options")],
    outputs=[gr.outputs.Image(type="pil",label='Deep subtraction'), gr.outputs.Image(type="pil", label='Vessel segmentation')],
    title=title,
    description=description,
    article=article,
    examples=examples
)

demo.launch(server_name='0.0.0.0')

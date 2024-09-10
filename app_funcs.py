import numpy as np
import streamlit as st
from PIL import Image
from huggingface_hub import from_pretrained_keras
import torch
from PIL import Image
import numpy as np
from realesrgan import RealESRGANer
import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from tensorflow.keras.preprocessing.image import img_to_array
from helper import *
import gc
from nafnet.models import create_model
from nafnet.utils import img2tensor as _img2tensor, tensor2img, imwrite
from nafnet.utils.options import parse

@st.cache_data(show_spinner=False)
def instantiate_model():
    model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet", compile=False)

    return model

@st.cache_data(show_spinner=False)
def enhance_image(uploaded_image, downloaded_image):
    model = instantiate_model()
    low_light_img = Image.open(uploaded_image).convert('RGB')
    input_shape = low_light_img.size
    low_light_img = low_light_img.resize((600, 400), Image.NEAREST)

    image = img_to_array(low_light_img)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)

    output_image = output_image.reshape((np.shape(output_image)[0], np.shape(output_image)[1], 3))
    output_image = np.uint32(output_image)
    final_image = Image.fromarray(output_image.astype('uint8'), 'RGB')
    final_image = final_image.resize(input_shape)
    final_image.save(downloaded_image)

@st.cache_data(show_spinner=False)
def super_resolution(uploaded_image, types, output_dir="downloads"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if types=="Scale x2":
        model = CustomBSRGAN(model_path="/content/drive/MyDrive/models/BSRGANx2.pth",
                             hf_model=False, device=device, output_dir=output_dir)
        rst_image = model.predict(img_path=uploaded_image)
        return rst_image
    elif types=="Scale x4":
        model = CustomBSRGAN(model_path="/content/drive/MyDrive/models/BSRGAN.pth",
                             hf_model=False, device=device, output_dir=output_dir)
        rst_image = model.predict(img_path=uploaded_image)
        return rst_image

@st.cache_data(show_spinner=False)
def sr_real_esrgan(scale, input_path, types, face_enhance, output_path="downloads"):
    assert scale == 2 or scale ==4, "sclace only equal 2 or 4"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path=""
    scale = 2
    if types == "Scale x2":
        scale = 2
        model_path="/content/drive/MyDrive/models/RealESRGAN_x2plus.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    elif types == "Scale x4":
        scale = 4
        model_path = "/content/drive/MyDrive/models/RealESRGAN_x4plus.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    elif types == "Anime x4":
        scale = 4
        model_path = "/content/drive/MyDrive/models/RealESRGAN_x4plus_anime_6B.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    enhancer = RealESRGANer(model_path=model_path, scale=scale, model=model, device=device)

    face_enhancer = GFPGANer(
        # model_path='models\GFPGANv1.3.pth',
        model_path='/content/drive/MyDrive/models/GFPGANv1.3.pth',
        upscale=scale,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=enhancer,
        device=device)

    image_name=os.path.basename(input_path)
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if face_enhance:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        cv2.imwrite(f"{output_path}/enhanced_{image_name}", output)
    else:   
        output, _ = enhancer.enhance(img, outscale=4)
        cv2.imwrite(f"{output_path}/enhanced_{image_name}", output)
    
@st.cache_data(show_spinner=False)

def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

@st.cache_data(show_spinner=False)

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

@st.cache_data(show_spinner=False)

def NAFNetBlur(uploaded_image, output_path="downloads"):
    opt_path = 'options/test/REDS/NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)
    
    image_name = os.path.basename(uploaded_image)
    img_input = imread(uploaded_image)
    img = img2tensor(img_input)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()
    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()
    visuals = model.get_current_visuals()
    output = tensor2img([visuals['result']])
    cv2.imwrite(f"{output_path}/enhanced_{image_name}", output)

@st.cache_data(show_spinner=False)

def NAFNetNoise(uploaded_image, output_path="downloads"):
    opt_path = 'options/test/SIDD/NAFNet-width64.yml'
    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    model = create_model(opt)
    
    image_name = os.path.basename(uploaded_image)
    img_input = imread(uploaded_image)
    img = img2tensor(img_input)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()
    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()
    visuals = model.get_current_visuals()
    output = tensor2img([visuals['result']])
    cv2.imwrite(f"{output_path}/enhanced_{image_name}", output)
@st.cache_data(show_spinner=False)

def download_success():
    st.success('✅ Download Successful !!')

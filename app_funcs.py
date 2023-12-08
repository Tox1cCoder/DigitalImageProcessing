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


@st.cache_resource(show_spinner=False)
def instantiate_model():
    # model = tf.saved_model.load('MIRNet/')
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
def supper_resolution(model_path, uploaded_image, output_dir="downloads"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomBSRGAN(model_path=model_path, hf_model=False, device=device, output_dir=output_dir)
    rst_image = model.predict(img_path=uploaded_image)
    return rst_image

@st.cache_data(show_spinner=False)
def sr_real_esrgan(model_path, scale, input_path, output_path="downloads"):
    assert scale == 2 or scale ==4, "sclace only equal 2 or 4"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_x4 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    model_x2 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    enhancer = RealESRGANer(model_path=model_path, scale=scale, model=model_x4 if scale==4 else model_x2, device=device)
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
    # output, _ = enhancer.enhance(img, outscale=4)
    # the follow is for face enhancement
    _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
    cv2.imwrite(f"{output_path}/enhanced_{image_name}", output)


@st.cache_data(show_spinner=False)
def download_success():
    st.success('✅ Download Successful !!')

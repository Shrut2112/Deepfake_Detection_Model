import cv2
import numpy as np
from matplotlib import pyplot as plt
import streamlit as st
from PIL import Image
import io
from tensorflow.keras.models import load_model


model = load_model('Deepfake_classif_new.h5')

def compute_ela(img, quality=90):
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, 'JPEG', quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)
    resaved = np.array(resaved)
    ela_map = np.abs(img.astype(np.float32) - resaved.astype(np.float32))
    ela_map = (ela_map / ela_map.max()) * 255
    ela_map = ela_map.astype(np.uint8)
    return ela_map

def compute_dft_mag_phase(gray_img):
    f = np.float32(gray_img)
    dft = cv2.dft(f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    phase = cv2.phase(dft_shift[:,:,0], dft_shift[:,:,1])
    mag = np.log1p(mag)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    phase = (phase - phase.min()) / (phase.max() - phase.min() + 1e-8)
    return mag, phase


def get_dft_rgb_feat(image):
    
    img_up = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb_images = np.expand_dims(img_up, axis=0)  

    img_small = cv2.resize(image, (32, 32))
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    mag, phase = compute_dft_mag_phase(gray)
    ela_map = compute_ela(img_small)
    ela_chan = cv2.cvtColor(ela_map, cv2.COLOR_RGB2GRAY) / 255.0

    feat = np.stack([mag, phase, ela_chan], axis=-1)
    dft_data = np.expand_dims(feat, axis=0) 

    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(img_up)
    axs[0].set_title("Input Image")
    axs[0].axis("off")

    axs[1].imshow(mag, cmap='gray')
    axs[1].set_title("DFT Magnitude")
    axs[1].axis("off")

    axs[2].imshow(phase, cmap='gray')
    axs[2].set_title("DFT Phase")
    axs[2].axis("off")

    axs[3].imshow(ela_chan, cmap='gray')
    axs[3].set_title("ELA Map")
    axs[3].axis("off")

    st.pyplot(fig)

    return [rgb_images, dft_data]


def get_prediction(input_pair):
    """Make model prediction from [rgb_input, dft_input]."""
    pred_prob = model.predict(input_pair)
    return pred_prob
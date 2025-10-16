import os
from pathlib import Path
from typing import List

import streamlit as st
from PIL import Image

from src.utils import load_pipe, read_config


st.set_page_config(page_title="LoRA SD Generator", page_icon="🎨", layout="wide")
st.title("Stable Diffusion + LoRA — Image Generator")

with st.sidebar:
    st.header("Settings")
    cfg = read_config()
    default_device = "cuda"
    lora_path = st.text_input("LoRA path", value=str(Path("outputs/lora").resolve()))
    device = st.selectbox("Device", options=["cuda", "cpu", "mps"], index=0)
    num_images = st.number_input("Num images", min_value=1, max_value=16, value=4, step=1)
    steps = st.number_input("Inference steps", min_value=10, max_value=75, value=35, step=5)
    generate_btn = st.button("Generate")

prompt = st.text_area("Prompt", height=100, placeholder="A photo of a cute cat, 4k, ultra detailed")


@st.cache_resource(show_spinner=True)
def get_pipe_cached(lora_path: str, device: str):
    return load_pipe(lora_path=lora_path, device=device)


cols = st.columns(4)

if generate_btn:
    if not prompt.strip():
        st.error("Prompt cannot be empty")
    else:
        try:
            pipe = get_pipe_cached(lora_path, device)
            images: List[Image.Image] = []
            with st.spinner("Generating images..."):
                for _ in range(num_images):
                    img = pipe(prompt, num_inference_steps=int(steps)).images[0]
                    images.append(img)

            for i, img in enumerate(images):
                cols[i % 4].image(img, caption=f"img_{i}", use_container_width=True)
        except Exception as e:
            st.exception(e)



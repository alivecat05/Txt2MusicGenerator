import streamlit as st  # 导入 Streamlit，用于构建 Web 应用
import os  # 用于文件路径操作
import torch  # 用于处理张量
import torchaudio  # 用于音频处理
import numpy as np  # 导入 NumPy
import base64  # 用于处理二进制数据的 Base64 编码
from audiocraft.models import MusicGen  # 导入 MusicGen 模型

# 定义模型路径

# 缓存资源，避免重复加载模型
@st.cache_resource
def load_model():
    # 加载预训练的 MusicGen 模型
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    return model

# 生成音乐张量函数
def generate_music_tensors(description, duration: int):
    print("Description", description)
    print("duration", duration)
    
    # 加载缓存的模型
    model = load_model()
    
    # 设置模型生成参数
    model.set_generation_params(use_sampling=True, top_k=250, duration=duration)
    
    # 生成音乐，返回结果张量
    output = model.generate(descriptions=[description])
    
    return output[0]

# 保存音频文件函数
def save_audio(samples: torch.Tensor):
    sample_rate = 32000  # 设置采样率
    save_path = "AudioGenerator/Audio_Generator/audio_ouput"  # 保存音频文件的路径
    
    # 如果保存路径不存在，创建它
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # 检查张量维度
    assert samples.dim() == 2 or samples.dim() == 3
    
    # 将张量移至 CPU 以便保存
    samples = samples.detach().cpu()
    
    # 如果张量是 2D，则增加一个维度
    if samples.dim() == 2:
        samples = samples[None, ...]
    
    # 遍历并保存每个音频样本
    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)
        return audio_path  # 返回保存的音频路径

# 生成音频文件的下载链接
def get_binary_file_html(bin_file, file_label='File'):
    # 打开并读取二进制文件
    with open(bin_file, 'rb') as f:
        data = f.read()
    
    # 对二进制数据进行 Base64 编码
    bin_str = base64.b64encode(data).decode()
    
    # 生成带有下载链接的 HTML 代码
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    
    return href

def style_recommand():
    st.write("""
    ### Classical Music
    - **Baroque**
    - **Classical**
    - **Romantic**
    - **Impressionism**
    - **Modern**
    - **Postmodern**

    ### Pop Music
    - **Pop**
    - **Rock**
    - **Pop Rock**
    - **Dance Pop**
    - **Indie Pop**
    - **Country Pop**

    ### Electronic Music
    - **EDM**
    - **House**
    - **Techno**
    - **Drum and Bass**
    - **Ambient**
    - **Synth Pop**

    ### Experimental Music
    - **Noise Music**
    - **Minimalism**
    - **Post-Industrial**
    - **Avant-Garde**
    - **Electroacoustic**

    ### Ambient Music
    - **Dark Ambient**
    - **Space Ambient**
    - **Meditative Ambient**
    - **New Age**
    - **Field Recording**
    - **Drone**
    """)
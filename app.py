import streamlit as st  # 导入 Streamlit，用于构建 Web 应用
import audio_Gen
import torch
from translate import Translator
import warnings
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置环境变量以管理 CUDA 内存
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 忽略指定的警告
warnings.filterwarnings("ignore", category=UserWarning, message="torch.nn.utils.weight_norm is deprecated")
# 设置 CUDA 内存占用
torch.cuda.set_per_process_memory_fraction(0.5, device=0)
def initialize_session_state():
    """初始化 Streamlit 会话状态中的所有属性。"""
    if 'rag_mode' not in st.session_state:
        st.session_state['rag_mode'] = False
    if 'flag' not in st.session_state:
        st.session_state['flag'] = 0
    if 'newprompt' not in st.session_state:
        st.session_state['newprompt'] = ""

def translate_text(text, src='zh', dest='en'):
    """使用 translate 库进行翻译，处理可能的错误。"""
    translator = Translator(from_lang=src, to_lang=dest)
    try:
        translated_text = translator.translate(text)
        return translated_text
    except Exception as e:
        st.error(f"翻译时出现错误: {e}")
        return text
def main():
    # 初始化 Streamlit 页面配置
    st.set_page_config(page_icon=":musical_note:", page_title="Music Generator")
    # 初始化会话状态
    initialize_session_state()
    # 网站大标题
    st.title("Text2Music音频生成器")
    # 添加可扩展的解释区域
    with st.expander("介绍"):
        st.write("Text2Music是一款基于audiocraft音频模型与浪潮源大模型的根据提示词生成音频的应用, 您可以在下方输入想要的声音, 稍等片刻就会有惊艳的声音给到你。")
    # 用户输入区域，用于输入描述文本
    text_area = st.text_area("请输入提示词：", value="你可以输入像'水流的声音,鸟叫,爵士乐曲,钢琴,吉他'等等", key="prompt")
    # 定义RAG ON按钮
    if st.button("选择风格", key='start'):
        # 切换RAG模式
        st.session_state['rag_mode'] = not st.session_state['rag_mode']
    # 如果RAG模式被激活，显示输入区域
    if st.session_state['rag_mode']:
        theme = st.text_area("请输入风格：", key="style", disabled=False)
        with st.expander("推荐输入"):
             audio_Gen.style_recommand()
        if st.button("StyleGO", key='Go'):
            import RAG
            st.session_state['newprompt'] = RAG.prompt_enhance_RAG(text_area, theme)
            st.session_state['flag'] = 1  # 设置标志位用于其他逻辑处理
            st.success("RAG Done!")
    # 选择生成时长的滑块
    time_slider = st.slider("选择生成时长（秒）", 2, 20, 20)

    if st.button("Generate!!!!", key='Generate'):
        st.session_state['flag'] = 1
    # 根据标志位选择提示词
    if st.session_state['flag'] == 1:
        newprompt = st.session_state['newprompt'] if st.session_state['rag_mode'] else text_area
        print(f'textOG: {newprompt}')
        translated_text = translate_text(newprompt)
        print(f'textNOW: {translated_text}')
        
        if translated_text and time_slider:
            # 显示用户输入的描述和时长
            st.json(
                {
                    "描述": translated_text,
                    "时长": time_slider
                }
            )
            # 创建占位符
            status_placeholder = st.empty()
            status_placeholder.subheader("正在生成音乐...")
            torch.cuda.empty_cache()
            # 生成音乐
            music_tensor = audio_Gen.generate_music_tensors(translated_text, time_slider)
            print("Music Tensors:", music_tensor)

            # 保存生成的音乐文件
            audio_filepath = audio_Gen.save_audio(music_tensor)
            torch.cuda.empty_cache()
            # 在 Streamlit 中播放音频
            with open(audio_filepath, 'rb') as audio_file:
                audio_byte = audio_file.read()
                st.audio(audio_byte)
            # 显示下载链接，允许用户下载生成的音频文件
            st.markdown(audio_Gen.get_binary_file_html(audio_filepath, '音频文件'), unsafe_allow_html=True)
            # 更新占位符内容为“完成！”
            status_placeholder.subheader("完成！")
            # 清理生成过程中的显存
            torch.cuda.empty_cache()

# 运行主函数
if __name__ == "__main__":
    main()

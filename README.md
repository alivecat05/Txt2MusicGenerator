# Txt2MusicGenerator
This is an audio generation application based on the LangChao Yuan 2.0 large model, integrating the MusicGen model and RAG (Retrieval-Augmented Generation) technology to optimize prompts.
You can enter any text into the text area and select from various music styles, allowing the app to generate more tailored and captivating music for you.
#install lib
open the terminal and type in
pip install -r requirements.txt
#install Yuan2.0
git lfs install
git clone https://www.modelscope.cn/IEITYuan/Yuan2-2B-Mars-hf.git
#install MusicGen
python -m pip install 'torch==2.1.0'
python -m pip install setuptools wheel
python -m pip install -U audiocraft
python -m pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft
python -m pip install -e .
python -m pip install -e '.[wm]'

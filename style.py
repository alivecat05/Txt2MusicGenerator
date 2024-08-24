from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
import gc
from translate import Translator
import streamlit as st
import RAG
# 定义大语言模型类
class LLM:
    """
    Class for Yuan2.0 LLM
    """
    def __init__(self, model_path: str) -> None:
        print("Create tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
        self.tokenizer.add_tokens(['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>', '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>', '<jupyter_code>', '<jupyter_output>', '<empty_output>'], special_tokens=True)
        print("Create model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to('cuda:0')
        print(f'Loading Yuan2.0 model from {model_path}.')
    def generate(self, style:str, content:str) -> str:
        # 释放未使用的内存
        prompt = f'介绍一下{style}音乐风格的{content}'
        torch.cuda.empty_cache()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to('cuda:0')

        if self.tokenizer.pad_token_id is not None:
            attention_mask = torch.ne(input_ids, self.tokenizer.pad_token_id).long().to('cuda:0')
        else:
            attention_mask = torch.ones_like(input_ids).to('cuda:0')
        # 使用 torch.no_grad() 禁用梯度计算
        with torch.no_grad():
            with torch.cuda.amp.autocast():  # 自动混合精度
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_length=1024,  # 减少生成文本的长度
                    temperature=0.5,
                    top_k=10,  # 控制多样性
                    top_p=0.2   # 控制多样性
                )
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取实际描述内容（假设分隔符是<sep>）
        if "<sep>" in output:
            description = output.split("<sep>", 1)[1].strip()
        else:
            description = output.strip()

        # 去除与提示重复的部分
        prompt_text = f'介绍一下{style}音乐风格的{content}'
        if description.startswith(prompt_text):
            description = description[len(prompt_text):].strip()

        torch.cuda.empty_cache()  # 释放显存
        return description

print("> Create Yuan2.0 LLM...")
model_path = './Yuan2-2B-Mars-hf'
llm = LLM(model_path)
document_path = "./knowledgeStyle.txt"
def getMusicStyle(style: str) -> str:
    index = RAG.VectorStoreIndex(document_path, RAG.embed_model)
    _style = index.query(style)
    content = "风格特点,代表人物，代表作品"
    outputs = llm.generate(_style, content)

    return outputs



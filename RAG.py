from typing import List
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from modelscope import snapshot_download
import gc

# 下载和加载嵌入模型
model_dir = snapshot_download("AI-ModelScope/bge-small-en-v1.5", cache_dir='.')

class EmbeddingModel:
    """
    Class for EmbeddingModel
    """
    def __init__(self, path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, legacy=False)
        self.model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16, trust_remote_code=True).half().to('cuda:0')
        print(f'Loading EmbeddingModel from {path}.')

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Calculate embeddings for a list of texts
        """
        # 设置最大输入长度，避免超出显存限制
        max_length = 80
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        encoded_input = {k: v.to('cuda:0') for k, v in encoded_input.items()}
        with torch.no_grad():  # 禁用梯度计算以节省内存
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0].to('cuda:0')  # 保证张量在GPU上
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        torch.cuda.empty_cache()  # 释放显存
        return sentence_embeddings.cpu().tolist()  # 返回CPU上的结果以便进一步处理

print("> Create embedding model...")
embed_model_path = './AI-ModelScope/bge-small-en-v1.5'
embed_model = EmbeddingModel(embed_model_path)

# 定义向量库索引类
class VectorStoreIndex:
    """
    Class for VectorStoreIndex
    """
    def __init__(self, document_path: str, embed_model: EmbeddingModel) -> None:
        self.documents = [line.strip() for line in open(document_path, 'r', encoding='utf-8')]
        self.embed_model = embed_model
        self.vectors = self.embed_model.get_embeddings(self.documents)
        print(f'Loading {len(self.documents)} documents for {document_path}.')
       
    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        """
        vector1 = torch.tensor(vector1, device='cuda:0')
        vector2 = torch.tensor(vector2, device='cuda:0')
        dot_product = torch.dot(vector1, vector2).item()
        magnitude = torch.norm(vector1) * torch.norm(vector2)
        return dot_product / magnitude.item() if magnitude != 0 else 0
        
    def query(self, theme: str, k: int = 1) -> List[str]:
        theme_vector = self.embed_model.get_embeddings([theme])[0]
        result = np.array([self.get_similarity(theme_vector, vector) for vector in self.vectors])
        sorted_indices = result.argsort()[-k:][::-1]
        return [self.documents[i] for i in sorted_indices]

print("> Create index...")
document_path = "./knowledge.txt"
index = VectorStoreIndex(document_path, embed_model)

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

    def generate(self, question: str, theme: List[str], temperature: float = 0.7) -> str:
        if theme:
            theme_text = " and ".join(theme)
            prompt = f'Describe a vivid and emotional description combining {question} with music style {theme_text}.'
        else:
            prompt = question

        prompt += "<sep>"

        # 释放未使用的内存
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
                    max_length=80,  # 减少生成文本的长度
                    temperature=temperature,
                    # top_k=30,  # 控制多样性
                    # top_p=0.5   # 控制多样性
                )

        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取实际描述内容（假设分隔符是<sep>）
        if "<sep>" in output:
            description = output.split("<sep>", 1)[1].strip()
        else:
            description = output.strip()

        # 去除与提示重复的部分
        prompt_text = f'Describe a vivid and emotional description combining {question} with music style {theme_text}.'
        if description.startswith(prompt_text):
            description = description[len(prompt_text):].strip()

        torch.cuda.empty_cache()  # 释放显存
        return description

print("> Create Yuan2.0 LLM...")
model_path = './Yuan2-2B-Mars-hf'
llm = LLM(model_path)

def prompt_enhance(question: str) -> str:
    return llm.generate(question, [], temperature=1.0)

def prompt_enhance_RAG(question: str, theme: str) -> str:
    # 从向量库中检索最相似的主题内容
    _theme = index.query(theme)
    result = llm.generate(question, _theme, temperature=1.0)
    gc.collect()
    torch.cuda.empty_cache()
    return result



if __name__=='__main__':
    rel = prompt_enhance_RAG("水流的声音","印象派")
    print("rel: ",rel)
    
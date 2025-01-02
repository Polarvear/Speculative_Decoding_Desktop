import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 초안 모델(Mq) 로드
def load_mq_model():
    mq_model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(mq_model_name)
    model = AutoModelForCausalLM.from_pretrained(mq_model_name)
    model.eval()
    return tokenizer, model

# 목표 모델(Mp) 로드
def load_mp_model():
    mp_model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(mp_model_name)
    model = AutoModelForCausalLM.from_pretrained(mp_model_name)
    model.eval()
    return tokenizer, model

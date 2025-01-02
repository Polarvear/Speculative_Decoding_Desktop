import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# 1. 모델 및 토크나이저 로드
# -------------------------------

# 초안 모델 (Mq): DistilGPT-2 (경량 모델)
mq_model_name = "distilgpt2"
mq_tokenizer = AutoTokenizer.from_pretrained(mq_model_name)
mq_model = AutoModelForCausalLM.from_pretrained(mq_model_name)
mq_model.eval()

# 목표 모델 (Mp): GPT-Neo 1.3B (대형 모델)
mp_model_name = "EleutherAI/gpt-neo-1.3B"
mp_tokenizer = AutoTokenizer.from_pretrained(mp_model_name)
mp_model = AutoModelForCausalLM.from_pretrained(mp_model_name)
mp_model.eval()

# -------------------------------
# 2. Speculative Decoding 함수
# -------------------------------

def speculative_decoding(input_text, max_length=50, num_draft=3):
    """
    Speculative Decoding 구현:
    1. 초안 모델(Mq)이 후보 텍스트를 생성.
    2. 목표 모델(Mp)이 초안을 검증하고 최종 출력을 선택.
    
    Args:
        input_text (str): 입력 텍스트.
        max_length (int): 생성 텍스트 최대 길이.
        num_draft (int): 초안 후보 개수.

    Returns:
        str: 목표 모델(Mp)의 최종 선택 텍스트.
    """
    # ---------------------------
    # 1. 초안 모델(Mq)로 초안 생성
    # ---------------------------
    input_ids = mq_tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        draft_outputs = mq_model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_draft,
            do_sample=True,  # 랜덤 샘플링 활성화
            top_k=50         # top-k 샘플링
        )
    # 초안 텍스트 리스트 생성
    draft_sentences = [mq_tokenizer.decode(output, skip_special_tokens=True) for output in draft_outputs]

    # ---------------------------
    # 2. 목표 모델(Mp)로 초안 검증
    # ---------------------------
    best_score = float("-inf")
    best_sentence = None

    for sentence in draft_sentences:
        input_ids = mp_tokenizer(sentence, return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = mp_model(input_ids)
            logits = outputs.logits[:, -1, :]  # 마지막 토큰의 로짓값
            # 확률값 계산 (Softmax)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            score = probs.max().item()  # 가장 높은 확률값
        # 최적 초안 업데이트
        if score > best_score:
            best_score = score
            best_sentence = sentence

    return best_sentence

# -------------------------------
# 3. Speculative Decoding 테스트
# -------------------------------

if __name__ == "__main__":
    input_text = "The future of artificial intelligence is"
    print("Input:", input_text)

    # Speculative Decoding 실행
    final_output = speculative_decoding(input_text, max_length=50, num_draft=3)
    print("Final Output:", final_output)

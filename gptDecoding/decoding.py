import torch
import time

# Speculative Decoding 함수
def speculative_decoding(input_text, mq_model, mq_tokenizer, mp_model, mp_tokenizer, max_length=50, num_draft=3):
    start_time = time.time()  # 시작 시간 기록

    # 1. 초안 생성 (Mq)
    input_ids = mq_tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        draft_outputs = mq_model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_draft,
            do_sample=True,
            top_k=50
        )
    draft_sentences = [mq_tokenizer.decode(output, skip_special_tokens=True) for output in draft_outputs]

    # 2. 초안 검증 (Mp)
    best_score = float("-inf")
    best_sentence = None
    slm_tokens_used = 0  # SLM 성공적으로 사용된 토큰 수

    for sentence in draft_sentences:
        input_ids = mp_tokenizer(sentence, return_tensors="pt").input_ids
        slm_tokens_used += len(input_ids[0])  # 초안의 토큰 수 기록
        with torch.no_grad():
            outputs = mp_model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            score = probs.max().item()
        if score > best_score:
            best_score = score
            best_sentence = sentence

    end_time = time.time()  # 종료 시간 기록
    latency = end_time - start_time  # Latency 계산

    return best_sentence, slm_tokens_used, len(draft_sentences), latency

#기존 LLM(Mp) 단독으로 텍스트 생성.
def generate_with_llm(input_text, mp_model, mp_tokenizer, max_length=50):
    """
    기존 LLM(Mp) 단독으로 텍스트 생성.
    Args:
        input_text (str): 입력 텍스트.
        mp_model: 목표 모델 (LLM).
        mp_tokenizer: 목표 모델의 토크나이저.
        max_length (int): 생성 텍스트 최대 길이.

    Returns:
        str: 생성된 텍스트.
        float: Latency (생성 시간).
    """
    import time
    start_time = time.time()  # 시작 시간 기록

    input_ids = mp_tokenizer(input_text, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = mp_model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            top_k=50
        )
    end_time = time.time()  # 종료 시간 기록

    generated_text = mp_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    latency = end_time - start_time

    return generated_text, latency


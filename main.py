from transformers import AutoTokenizer, AutoModelForCausalLM
import torch  # 이 줄을 추가

# DistilGPT-2 초안 모델
draft_model_name = "distilgpt2"
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

# 초안 생성
input_text = "The future of AI is"
input_ids = draft_tokenizer(input_text, return_tensors="pt").input_ids

# 여러 초안 생성
draft_outputs = draft_model.generate(
    input_ids, 
    max_length=50, 
    num_return_sequences=3, 
    do_sample=True, 
    top_k=50
)
draft_sentences = [draft_tokenizer.decode(output, skip_special_tokens=True) for output in draft_outputs]
print("Draft Sentences:")
print("\n".join(draft_sentences))

# for i, seq in enumerate(draft_outputs):
#     print(f"Generated {i + 1}: {draft_tokenizer.decode(seq, skip_special_tokens=True)}")

# LLaMA 또는 Falcon 대형 모델
mp_model_name = "EleutherAI/gpt-neo-1.3B"  # 더 큰 모델
mp_tokenizer = AutoTokenizer.from_pretrained(mp_model_name)
mp_model = AutoModelForCausalLM.from_pretrained(mp_model_name)

# 각 초안을 검증 (logit 값 계산)
scores = []
for sentence in draft_sentences:
    input_ids = mp_tokenizer(sentence, return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = mp_model(input_ids)
        logit_score = outputs.logits[:, -1, :].mean().item()  # 간단히 평균 점수 계산
        scores.append((sentence, logit_score))

# 점수에 따라 최적 초안 선택
best_sentence = max(scores, key=lambda x: x[1])
print(f"Best Sentence: {best_sentence[0]} with score: {best_sentence[1]}")

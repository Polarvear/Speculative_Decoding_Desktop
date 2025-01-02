# from models import load_mq_model, load_mp_model
# from decoding import speculative_decoding
# from metrics import calculate_tur, calculate_latency

# if __name__ == "__main__":
#     # 모델 로드
#     mq_tokenizer, mq_model = load_mq_model()
#     mp_tokenizer, mp_model = load_mp_model()

#     # 입력 텍스트
#     input_text = "The future of artificial intelligence is"
#     print("Input:", input_text)

#     # Speculative Decoding 실행
#     final_output, slm_tokens_used, total_tokens, latency = speculative_decoding(
#         input_text, mq_model, mq_tokenizer, mp_model, mp_tokenizer, max_length=50, num_draft=3
#     )

#     # Token Utilization Rate 계산
#     tur = calculate_tur(slm_tokens_used, total_tokens)
#     print(f"Token Utilization Rate: {tur:.2%}")

#     # Latency 계산
#     latency_per_token = calculate_latency(latency, total_tokens)
#     print(f"Latency per Token: {latency_per_token:.4f} seconds/token")

#     # 최종 출력
#     print("Final Output:", final_output)



from models import load_mq_model, load_mp_model
from decoding import speculative_decoding, generate_with_llm
from metrics import calculate_tur, calculate_latency

if __name__ == "__main__":
    # 모델 로드
    mq_tokenizer, mq_model = load_mq_model()
    mp_tokenizer, mp_model = load_mp_model()

    # 입력 텍스트
    input_text = "The future of artificial intelligence is"
    print("Input:", input_text)

    # -------------------------
    # 1. Speculative Decoding
    # -------------------------
    final_output_sd, slm_tokens_used, total_tokens, latency_sd = speculative_decoding(
        input_text, mq_model, mq_tokenizer, mp_model, mp_tokenizer, max_length=50, num_draft=3
    )

    # Token Utilization Rate 계산
    tur_sd = calculate_tur(slm_tokens_used, total_tokens)
    latency_per_token_sd = calculate_latency(latency_sd, total_tokens)

    print("\n[Speculative Decoding]")
    print(f"Final Output: {final_output_sd}")
    print(f"Token Utilization Rate: {tur_sd:.2%}")
    print(f"Latency per Token: {latency_per_token_sd:.4f} seconds/token")

    # -------------------------
    # 2. 기존 LLM 단독 실행
    # -------------------------
    final_output_llm, latency_llm = generate_with_llm(
        input_text, mp_model, mp_tokenizer, max_length=50
    )
    latency_per_token_llm = calculate_latency(latency_llm, total_tokens)

    print("\n[Existing LLM]")
    print(f"Final Output: {final_output_llm}")
    print(f"Latency: {latency_llm:.2f} seconds")
    print(f"Latency per Token: {latency_per_token_llm:.4f} seconds/token")

    # -------------------------
    # 3. 결과 비교
    # -------------------------
    print("\n[Comparison]")
    print(f"TUR (Speculative Decoding): {tur_sd:.2%}")
    print(f"Latency per Token (Speculative Decoding): {latency_per_token_sd:.4f} seconds/token")
    print(f"Latency per Token (Existing LLM): {latency_per_token_llm:.4f} seconds/token")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple

class SpeculativeDecoder:
    def __init__(self):
        # 작은 모델 (draft model) 초기화
        self.draft_model_name = "distilgpt2"
        self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_name)
        self.draft_model = AutoModelForCausalLM.from_pretrained(self.draft_model_name)
        
        # 큰 모델 (target model) 초기화
        self.target_model_name = "EleutherAI/gpt-neo-1.3B"
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_model_name)
        self.target_model = AutoModelForCausalLM.from_pretrained(self.target_model_name)
        
        # padding 토큰 설정
        self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
        self.target_tokenizer.pad_token = self.target_tokenizer.eos_token

    def draft_tokens(self, input_ids: torch.Tensor, n_tokens: int = 4) -> List[int]:
        """작은 모델로 토큰 추측"""
        with torch.no_grad():
            draft_outputs = self.draft_model.generate(
                input_ids,
                max_length=input_ids.shape[1] + n_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                pad_token_id=self.draft_tokenizer.pad_token_id
            )
        return draft_outputs[0][input_ids.shape[1]:].tolist()

    def verify_tokens(self, input_ids: torch.Tensor, draft_tokens: List[int]) -> Tuple[List[int], int]:
        """큰 모델로 추측된 토큰 검증"""
        with torch.no_grad():
            # 입력 시퀀스에 대한 타겟 모델의 예측
            target_outputs = self.target_model(input_ids)
            target_logits = target_outputs.logits[0, -1, :]
            
            # 가장 가능성 높은 토큰
            target_next_token = torch.argmax(target_logits).item()
            
            # draft 토큰과 비교
            accepted_tokens = []
            for draft_token in draft_tokens:
                if draft_token == target_next_token:
                    accepted_tokens.append(draft_token)
                else:
                    break
                    
            return accepted_tokens, len(accepted_tokens)

    def generate(self, prompt: str, max_length: int = 50) -> str:
        """Speculative Decoding 수행"""
        input_ids = self.draft_tokenizer.encode(prompt, return_tensors="pt")
        generated_ids = input_ids[0].tolist()
        
        while len(generated_ids) < max_length:
            # 현재까지의 토큰으로 입력 생성
            current_input = torch.tensor([generated_ids])
            
            # 작은 모델로 다음 토큰들 추측
            draft_tokens = self.draft_tokens(current_input)
            if not draft_tokens:
                break
                
            # 큰 모델로 추측된 토큰 검증
            accepted_tokens, n_accepted = self.verify_tokens(current_input, draft_tokens)
            
            # 검증된 토큰 추가
            if n_accepted > 0:
                generated_ids.extend(accepted_tokens)
            else:
                # 추측이 틀린 경우 큰 모델로 한 토큰 생성
                with torch.no_grad():
                    target_outputs = self.target_model.generate(
                        current_input,
                        max_length=len(generated_ids) + 1,
                        do_sample=True,
                        top_k=50,
                        num_return_sequences=1,
                        pad_token_id=self.target_tokenizer.pad_token_id
                    )
                next_token = target_outputs[0][-1].item()
                generated_ids.append(next_token)
                
        return self.target_tokenizer.decode(generated_ids)

# 사용 예시
if __name__ == "__main__":
    decoder = SpeculativeDecoder()
    prompt = "The future of AI is"
    
    print("입력 프롬프트:", prompt)
    print("\n생성 중...")
    
    generated_text = decoder.generate(prompt)
    print("\n생성된 텍스트:")
    print(generated_text)
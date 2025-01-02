import torch
import torch.quantization
import gc
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import time

class OptimizedSpeculativeDecoder:
    def __init__(self, use_quantization: bool = True):
        # CUDA 최적화 설정
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_quantization = use_quantization
        
        # 드래프트 모델 초기화
        self.draft_model_name = "distilgpt2"
        self.draft_tokenizer = AutoTokenizer.from_pretrained(
            self.draft_model_name,
            use_fast=True,
            model_max_length=512
        )
        self.draft_model = self._load_model(self.draft_model_name)
        
        # 타겟 모델 초기화
        self.target_model_name = "EleutherAI/gpt-neo-1.3B"
        self.target_tokenizer = AutoTokenizer.from_pretrained(
            self.target_model_name,
            use_fast=True,
            model_max_length=512
        )
        self.target_model = self._load_model(self.target_model_name)
        
        # 패딩 토큰 설정
        self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
        self.target_tokenizer.pad_token = self.target_tokenizer.eos_token
        
        self._optimize_memory()

    def _load_model(self, model_name: str) -> torch.nn.Module:
        """모델 로드 및 최적화"""
        print(f"\n{model_name} 로딩 중...")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )
        
        if self.use_quantization:
            print(f"{model_name} 양자화 중...")
            model = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv1d},
                dtype=torch.qint8
            )
        
        return model.to(self.device)

    def _optimize_memory(self):
        """메모리 최적화"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def monitor_resources(self):
        """리소스 사용량 모니터링"""
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            print(f"GPU 메모리 사용량: {gpu_memory:.2f}MB")
        
        print(f"CPU 메모리 사용량: {memory_usage:.2f}MB")

    def draft_tokens(self, input_ids: torch.Tensor, n_tokens: int = 4) -> List[int]:
        """드래프트 모델로 토큰 생성"""
        try:
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                draft_outputs = self.draft_model.generate(
                    input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                    max_length=input_ids.shape[1] + n_tokens,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.draft_tokenizer.pad_token_id,
                    use_cache=True
                )
            return draft_outputs[0][input_ids.shape[1]:].tolist()
        except RuntimeError as e:
            print(f"드래프트 토큰 생성 중 오류: {e}")
            self._optimize_memory()
            return []

    def verify_tokens(self, input_ids: torch.Tensor, draft_tokens: List[int]) -> Tuple[List[int], int]:
        """타겟 모델로 토큰 검증"""
        try:
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                target_outputs = self.target_model(
                    input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device)
                )
                target_logits = target_outputs.logits[0, -1, :]
                target_next_token = torch.argmax(target_logits).item()
                
                accepted_tokens = []
                for draft_token in draft_tokens:
                    if draft_token == target_next_token:
                        accepted_tokens.append(draft_token)
                    else:
                        break
                
                return accepted_tokens, len(accepted_tokens)
        except RuntimeError as e:
            print(f"토큰 검증 중 오류: {e}")
            self._optimize_memory()
            return [], 0

    def generate(self, prompt: str, max_length: int = 50, batch_size: int = 1) -> str:
        """텍스트 생성"""
        try:
            print("\n텍스트 생성 시작...")
            start_time = time.time()
            
            input_ids = self.draft_tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids[:batch_size].to(self.device)
            generated_ids = input_ids[0].tolist()
            
            tokens_generated = 0
            while len(generated_ids) < max_length:
                current_input = torch.tensor([generated_ids]).to(self.device)
                
                # 드래프트 토큰 생성
                draft_tokens = self.draft_tokens(current_input)
                if not draft_tokens:
                    break
                
                # 토큰 검증
                accepted_tokens, n_accepted = self.verify_tokens(current_input, draft_tokens)
                
                if n_accepted > 0:
                    generated_ids.extend(accepted_tokens)
                    tokens_generated += n_accepted
                else:
                    # 단일 토큰 생성
                    attention_mask = torch.ones_like(current_input)
                    with torch.no_grad():
                        target_outputs = self.target_model.generate(
                            current_input,
                            attention_mask=attention_mask,
                            max_length=len(generated_ids) + 1,
                            do_sample=True,
                            top_k=50,
                            num_return_sequences=1,
                            pad_token_id=self.target_tokenizer.pad_token_id,
                            use_cache=True
                        )
                    next_token = target_outputs[0][-1].item()
                    generated_ids.append(next_token)
                    tokens_generated += 1
                
                # 메모리 최적화
                if len(generated_ids) % 10 == 0:
                    self._optimize_memory()
            
            generation_time = time.time() - start_time
            print(f"\n생성 완료!")
            print(f"생성 시간: {generation_time:.2f}초")
            print(f"생성된 토큰 수: {tokens_generated}")
            print(f"초당 토큰 생성: {tokens_generated/generation_time:.2f}")
            
            return self.target_tokenizer.decode(generated_ids)
            
        except Exception as e:
            print(f"텍스트 생성 중 오류 발생: {e}")
            self._optimize_memory()
            return prompt

def main():
    print("=== Optimized Speculative Decoder 초기화 중 ===")
    decoder = OptimizedSpeculativeDecoder(use_quantization=True)
    
    print("\n=== 초기 리소스 사용량 ===")
    decoder.monitor_resources()
    
    prompt = "The future of AI is"
    print(f"\n입력 프롬프트: {prompt}")
    
    generated_text = decoder.generate(prompt, max_length=100)
    
    print("\n=== 최종 리소스 사용량 ===")
    decoder.monitor_resources()
    
    print("\n=== 생성된 텍스트 ===")
    print(generated_text)

if __name__ == "__main__":
    main()
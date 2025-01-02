def calculate_tur(slm_tokens_used, total_tokens):
    """
    Token Utilization Rate 계산
    """
    return slm_tokens_used / total_tokens

def calculate_latency(latency, total_tokens):
    """
    Latency 계산
    """
    return latency / total_tokens

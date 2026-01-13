# make_prompt_128.py
# Usage:
#   python make_prompt_128.py meta-llama/Llama-2-7b-hf
#   python make_prompt_128.py gpt2

import sys
from transformers import AutoTokenizer

def make_prompt_exact_tokens(tokenizer, n_tokens=128, base=" hello", add_special_tokens=False):
    # 1) 충분히 긴 텍스트 만든 뒤 토큰으로 자르기
    text = base * (n_tokens * 8)
    ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)

    # 혹시 모자라면 더 붙이기 (드물지만 안전장치)
    while len(ids) < n_tokens:
        text += base * (n_tokens * 4)
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)

    ids = ids[:n_tokens]

    # 2) 토큰 -> 문자열 (decode)
    prompt = tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # 3) 검증: 재토큰화해서 정확히 n_tokens인지 확인 (틀리면 토큰공간에서 보정)
    check = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if len(check) != n_tokens:
        prompt = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        check = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

    assert len(check) == n_tokens, f"Expected {n_tokens} tokens, got {len(check)}"
    return prompt

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_prompt_128.py <model_name_or_path>")
        sys.exit(1)

    model_name = sys.argv[1]
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    prompt = make_prompt_exact_tokens(tok, n_tokens=128, base=" hello", add_special_tokens=False)

    print("=== PROMPT (first 200 chars) ===")
    print(repr(prompt[:200]))
    print("\n=== TOKEN LEN CHECK ===")
    print(len(tok.encode(prompt, add_special_tokens=False)))

if __name__ == "__main__":
    main()
import pandas as pd
from transformers import AutoTokenizer

# ✅ 설정
csv_path = "your_text_data_path"  # 원본 CSV 경로
output_path = "output_path"  # 저장 경로
model_name = "meta-llama/Llama-3.1-1B"
split_size = 128
min_length = 64
max_length = 8192  # LLaMA 3.1은 이 정도까지 지원

# ✅ Hugging Face Access Token (필요 시)
hf_token = "hf_your_huggingface_tokens"  # ← 네 토큰 입력

# ✅ 데이터 로드
df = pd.read_csv(csv_path)
texts = df['text'].astype(str).tolist()
labels = df['label'].tolist()

# ✅ 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=hf_token,
    trust_remote_code=True
)
tokenizer.model_max_length = max_length
tokenizer.truncation_side = 'right'

# ✅ 분할 결과 저장용 리스트
new_texts, new_labels = [], []

for text, label in zip(texts, labels):
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)

    for i in range(0, len(tokens), split_size):
        chunk = tokens[i:i+split_size]
        if len(chunk) >= min_length:
            decoded = tokenizer.decode(chunk, skip_special_tokens=True)
            new_texts.append(decoded)
            new_labels.append(label)

# ✅ 결과 저장
new_df = pd.DataFrame({'text': new_texts, 'label': new_labels})
new_df.to_csv(output_path, index=False)
print(f"✅ 저장 완료: {output_path}, 총 {len(new_df)} 샘플 생성")


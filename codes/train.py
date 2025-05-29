import os, gc, torch, pandas as pd, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from peft import get_peft_model, LoraConfig, TaskType
from tqdm.auto import tqdm
import torch.nn.functional as F

# ✅ 환경 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache(); gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 설정
train_path = "train.csv"     # 학습 데이터 경로
test_path = "test.csv"       # 테스트 데이터 경로
hf_token = "hf_your_token"   # Hugging Face 토큰
max_length = 128
batch_size = 1
accumulation_steps = 4
num_epochs = 3
learning_rate = 2e-5

# ✅ Dataset 정의
class TextDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.encodings["input_ids"])

# ✅ 모델 설정
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
output_dir = "./output_llama_8b_lora"
torch_dtype = torch.float32

lora_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.1,
    task_type=TaskType.SEQ_CLS, bias="none"
)

# ✅ 학습 및 추론 함수
def train_and_save():
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path).dropna()

    train_enc = tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=max_length)
    test_enc = tokenizer(test_df["text"].tolist(), truncation=True, padding=True, max_length=max_length)

    train_dataset = TextDataset(train_enc, df["label"].tolist())
    test_dataset = TextDataset(test_enc)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch_dtype,
        token=hf_token
    ).to(device)

    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader) * num_epochs // accumulation_steps
    )

    # ✅ 학습 루프
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = F.cross_entropy(outputs.logits, batch["labels"]) / accumulation_steps
            loss.backward()
            total_loss += loss.item()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        print(f"✅ Epoch {epoch+1} 완료 - Loss: {total_loss:.4f}")

    # ✅ 추론
    model.eval()
    softmax_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="추론 중"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = F.softmax(outputs.logits, dim=1)[:, 1]
            softmax_probs.extend(probs.cpu().numpy())

    # ✅ 결과 저장
    np.save(os.path.join(output_dir, "softmax.npy"), np.array(softmax_probs))
    print(f"✅ Softmax 결과 저장 완료: {output_dir}")

    # ✅ 서브미션 생성
    submission = pd.read_csv(test_path)
    submission["prediction"] = (np.array(softmax_probs) > 0.5).astype(int)
    submission[["id", "prediction"]].to_csv(
        os.path.join(output_dir, "submission.csv"), index=False
    )
    print(f"📄 서브미션 파일 저장 완료: {os.path.join(output_dir, 'submission.csv')}")

# ✅ 실행
train_and_save()

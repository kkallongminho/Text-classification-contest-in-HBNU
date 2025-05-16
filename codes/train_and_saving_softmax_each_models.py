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
train_path = "output_by_undersampling.py"
test_path = "Output_path"
hf_token = "hf_your_huggingface_tokens"
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

# ✅ 모델 구성 (float32 + LoRA)
model_configs = [
    {
        "name": "LLaMA-1B",
        "model_name": "meta-llama/Llama-3.2-1B",
        "output_dir": "./output_llama_1b_lora",
        "torch_dtype": torch.float32,
        "use_lora": True,
        "lora_config": LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1,
            task_type=TaskType.SEQ_CLS, bias="none"
        ),
        "loss_fn": torch.nn.CrossEntropyLoss()
    },
    {
        "name": "LLaMA-3B",
        "model_name": "meta-llama/Llama-3.1-3B",
        "output_dir": "./output_llama_3b_lora",
        "torch_dtype": torch.float32,
        "use_lora": True,
        "lora_config": LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1,
            task_type=TaskType.SEQ_CLS, bias="none"
        ),
        "loss_fn": torch.nn.CrossEntropyLoss()
    },
    {
        "name": "Mistral-7B",
        "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
        "output_dir": "./output_mistral_lora",
        "torch_dtype": torch.float32,
        "use_lora": True,
        "lora_config": LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.1,
            task_type=TaskType.SEQ_CLS, bias="none"
        ),
        "loss_fn": torch.nn.CrossEntropyLoss()
    }
]

# ✅ 학습 + 추론 함수
def train_and_save(config):
    print(f"\n🚀 학습 시작: {config['name']}")
    os.makedirs(config["output_dir"], exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"], use_fast=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    df = pd.read_csv(train_path, encoding='utf-8')
    test_df = pd.read_csv(test_path, encoding='utf-8').dropna()

    train_enc = tokenizer(df["text"].tolist(), truncation=True, padding=True, max_length=max_length)
    test_enc = tokenizer(test_df["text"].tolist(), truncation=True, padding=True, max_length=max_length)

    train_dataset = TextDataset(train_enc, df["label"].tolist())
    test_dataset = TextDataset(test_enc)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=2,
        torch_dtype=config["torch_dtype"],
        token=hf_token
    ).to(device)

    if config["use_lora"]:
        model = get_peft_model(model, config["lora_config"])

    if "llama" in config["model_name"].lower():
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0,
        num_training_steps=len(train_loader) * num_epochs // accumulation_steps
    )

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = config["loss_fn"](outputs.logits, batch["labels"]) / accumulation_steps
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

    np.save(os.path.join(config["output_dir"], "softmax.npy"), np.array(softmax_probs))
    print(f"✅ {config['name']} softmax 저장 완료")

# ✅ 전체 실행
for config in model_configs:
    train_and_save(config)

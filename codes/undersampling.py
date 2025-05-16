import pandas as pd

# ✅ 경로 설정
csv_path = "output_"
output_path = "output_by_split_by_128tokens.py"

# ✅ CSV 로드 (줄바꿈 포함 대응)
df = pd.read_csv(csv_path, encoding="utf-8", engine="python", quotechar='"')

# ✅ 라벨 분리
label_0 = df[df["label"] == 0]
label_1 = df[df["label"] == 1]

# ✅ 최소 라벨 수에 맞춰 undersampling
min_count = min(len(label_0), len(label_1))
label_0_balanced = label_0.sample(n=min_count, random_state=42)
label_1_balanced = label_1.sample(n=min_count, random_state=42)

# ✅ 합치기 및 셔플
balanced_df = pd.concat([label_0_balanced, label_1_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)

# ✅ 저장
balanced_df.to_csv(output_path, index=False, encoding="utf-8")

print(f"✅ 저장 완료: {output_path}")
print(f"라벨별 개수:\n{balanced_df['label'].value_counts()}")


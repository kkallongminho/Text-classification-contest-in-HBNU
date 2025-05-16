import pandas as pd

# ✅ CSV 경로
csv_path = "output_by_split_by_128tokens.py"

# ✅ 줄바꿈 포함 데이터도 안전하게 로딩
df = pd.read_csv(csv_path, encoding="utf-8", engine="python", quotechar='"')

# ✅ 라벨별 개수 세기
label_counts = df["label"].value_counts()

# ✅ 출력
print("✅ 라벨별 샘플 수:")
print(label_counts)


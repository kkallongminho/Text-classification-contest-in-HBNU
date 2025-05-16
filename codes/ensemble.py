import numpy as np
import pandas as pd
import os

# ✅ softmax 결과 경로들
softmax_paths = [
    ".output_llama_1b_lora/softmax.npy",
    "./output_llama_3b_lora/softmax.npy",  # 사용 가능하면 추가
    "./output_llama_1b_lora/softmax.npy"
]

# ✅ test 데이터 ID 로드
test_csv_path = "output_path"
test_df = pd.read_csv(test_csv_path, encoding="utf-8")
ids = test_df["id"].tolist()

# ✅ softmax 앙상블
softmax_list = [np.load(path) for path in softmax_paths]
ensemble_softmax = np.mean(softmax_list, axis=0)

# ✅ threshold 적용 → label 생성
labels = (ensemble_softmax >= 0.5).astype(int)

# ✅ submission.csv 저장
submission = pd.DataFrame({"id": ids, "label": labels})
submission.to_csv("submission.csv", index=False)
print("✅ 앙상블 완료: submission.csv 생성됨")


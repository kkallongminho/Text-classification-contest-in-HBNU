
⸻

🧠 LLM-based Text Classification with LLaMA-3-8B and LoRA

This project performs binary text classification (real vs. AI-generated) using the Meta-Llama-3-8B-Instruct model. It focuses on efficient preprocessing, lightweight fine-tuning with LoRA, and final probability-based submission using softmax.

⸻

🚀 Features

✅ 1. Token-Based Preprocessing
	•	Splits text data into 128-token chunks
	•	Discards samples shorter than 64 tokens
	•	Maintains consistent input length for model stability

✅ 2. Undersampling for Class Balance
	•	Reduces the size of overrepresented class
	•	Ensures balanced training data distribution

✅ 3. Training with LoRA
	•	Uses Low-Rank Adaptation (LoRA) to fine-tune large models with reduced memory and computation cost
	•	Trains the LLaMA-3-8B model in float32 precision
	•	Utilizes CrossEntropyLoss with a warm-up learning rate scheduler

✅ 4. Inference with Softmax
	•	Computes softmax probabilities from logits
	•	Outputs the final .npy and .csv files for submission

⸻

🛠️ Requirements

pip install torch transformers peft accelerate

🧪 Usage Flow

split_by_128tokens.py 
→ count_amount_each_labels.py 
→ undersampling.py 
→ train_and_save_softmax_llama8b.py 
→ save_submission.py

📊 Tuning Tips
	•	Increase batch_size if your GPU memory allows
	•	Adjust accumulation_steps, learning_rate, and num_epochs for better performance

⸻

📌 Notes
	•	Hugging Face access token is required to load the LLaMA-3-8B model
	•	Small batch sizes are used to prevent out-of-memory errors
	•	LoRA enables efficient training of large-scale models even on limited hardware



![결과 이미지](스크린샷 2025-06-11 오후 5.14.37)

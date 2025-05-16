🧠 LLM-based Text Classification with Token Splitting, LoRA, and Softmax Ensemble

This project performs binary text classification (real vs. AI-generated) using transformer-based models like LLaMA 3.2-1B, LLaMA 3.1-3B, and Mistral-7B. It includes:
	•	Efficient data tokenization/splitting
	•	Class balancing via undersampling
	•	Training with LoRA (Low-Rank Adaptation)
	•	Softmax-based ensemble for final submission.

🚀 Features

✅ 1. Token-Based Splitting
	•	Texts are split into chunks of 128 tokens
	•	Discards samples shorter than 64 tokens
	•	Ensures uniform input size for transformer models

✅ 2. Undersampling
	•	Balances the dataset by reducing overrepresented class
	•	Shuffles after sampling to improve learning stability

✅ 3. Training with LoRA
	•	Applies Low-Rank Adaptation (LoRA) to reduce training cost
	•	Trains LLaMA and Mistral models in float32
	•	Uses CrossEntropyLoss with warmup scheduler

✅ 4. Inference & Ensemble
	•	Applies softmax to each model’s logits
	•	Averages multiple models’ outputs for robust final prediction
	•	Saves .npy and .csv outputs


🛠️ Requirements

pip install torch transformers datasets peft accelerate

🧪 Usage

1. Data Preprocessing

Split texts into 128-token segments and save them:

2. Train All Models

3. Softmax Ensemble + Submission

split_by_128tokens.py --> count_amount_each_labels.py --> undersampling.py --> train_and_saving_softmax_each_models.py --> ensemble.py



📊 Performance Tuning Tips
	•	Use larger batch sizes if memory allows
	•	Tune accumulation_steps, lr, and num_epochs
	•	Optionally experiment with BCEWithLogitsLoss for binary outputs


📌 Notes
	•	All models use Hugging Face access tokens for loading private models
	•	DataLoader uses small batch_size to avoid OOM issues on Colab
	•	LoRA significantly reduces memory usage for large models

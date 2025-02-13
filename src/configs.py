from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="task-specific-distilbert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
)

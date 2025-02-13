import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_preprocess_data(dataset_name="go_emotions", model_name="bert-large-cased"):
    dataset = load_dataset(dataset_name, "raw")["train"]
    label_names = dataset.column_names[10:]
    
    dataset = dataset.map(lambda x: {"labels": [label for label in label_names if x[label]]})
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
    
    unique_labels = sorted(set(sum(dataset["train"]["labels"], [])))
    id2label = {idx: label for idx, label in enumerate(unique_labels)}
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_data(examples):
        one_hot_labels = [
            [1 if l in labels else 0 for l in unique_labels] for labels in examples["labels"]
        ]
        encoding = tokenizer(examples["text"], truncation=True, max_length=256, padding="max_length")
        encoding["labels"] = one_hot_labels
        return encoding
    
    dataset = dataset.map(preprocess_data, batched=True, batch_size=128)
    return dataset, tokenizer, id2label, label2id
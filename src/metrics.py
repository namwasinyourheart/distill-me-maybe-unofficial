# from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, jaccard_score, precision_score, recall_score
# import torch
# import numpy as np

# def multi_label_metrics(predictions, labels, threshold=0.5):
#     sigmoid = torch.nn.Sigmoid()
#     probs = sigmoid(torch.tensor(predictions))
#     y_pred = (probs >= threshold).numpy().astype(int)
    
#     return {
#         "f1": f1_score(labels, y_pred, average="micro"),
#         "roc_auc": roc_auc_score(labels, y_pred, average="micro"),
#         "accuracy": accuracy_score(labels, y_pred),
#         "jaccard": jaccard_score(labels, y_pred, average="micro"),
#         "precision": precision_score(labels, y_pred, average="micro"),
#         "recall": recall_score(labels, y_pred, average="micro")
#     }

# def compute_metrics(p):
#     preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # return multi_label_metrics(preds, p.label_ids)
# 
import numpy as np
import evaluate
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_metric.compute(
        predictions=predictions, references=labels
    )
    return {
        "accuracy": acc["accuracy"],
    }

# accuracy_metric = evaluate.load("accuracy")
# results = accuracy_metric.compute(references=[0, 1], predictions=[0, 1])
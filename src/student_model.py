from transformers import AutoModelForSequenceClassification

def load_student_model(model_name="distilbert-base-cased", num_labels=10, id2label=None, label2id=None):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )

from transformers import AutoModelForSequenceClassification

def get_teacher_model(model_name, num_labels, id2label, label2id):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        problem_type="multi_label_classification",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    return model
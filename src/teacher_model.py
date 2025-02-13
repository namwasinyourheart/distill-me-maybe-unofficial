
from transformers import AutoModelForSequenceClassification

def load_teacher_model(model_name="teacher-bert"):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, problem_type="multi_label_classification"
    ).eval()

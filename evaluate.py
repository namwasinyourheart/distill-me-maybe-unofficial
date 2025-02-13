from train import train_teacher_model

def evaluate_model():
    trainer = train_teacher_model()
    results = trainer.evaluate()
    print("Evaluation Results:", results)
    return results

if __name__ == "__main__":
    evaluate_model()

# from student_model import load_student_model
# from prepare_data import load_dataset
# from transformers import Trainer
# from metrics import compute_metrics
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("teacher-bert")
# _, eval_dataset = load_dataset(tokenizer)

# model = load_student_model(num_labels=15)
# trainer = Trainer(model)

# eval_results = trainer.evaluate(eval_dataset)
# print(eval_results)

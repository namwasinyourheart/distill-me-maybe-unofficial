from transformers import AutoTokenizer

def load_tokenizer(data_args, model_args):
    teacher_tokenizer = AutoTokenizer.from_pretrained(model_args.teacher_model_name_or_path)
    student_tokenizer = AutoTokenizer.from_pretrained(model_args.student_model_name_or_path)

    sample = "Here's our sanity check."

    assert teacher_tokenizer(sample) == student_tokenizer(sample), (
        "Tokenizers need to have the same output! "
        f"{teacher_tokenizer(sample)} != {student_tokenizer(sample)}"
    )

    del teacher_tokenizer
    # del student_tokenizer

    # tokenizer = AutoTokenizer.from_pretrained(model_args.teacher_model_name_or_path)
    tokenizer = student_tokenizer
    
    if not tokenizer.pad_token:
        if data_args.tokenizer.new_pad_token:
            tokenizer.padding_side = 'left'
            tokenizer.pad_token = data_args.tokenizer.new_pad_token,
            tokenizer.add_special_tokens({"pad_token": data_args.tokenizer.new_pad_token})
        else:
            tokenizer.padding_side = 'right'
            tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer



from transformers import AutoModelForSequenceClassification

def load_model(model_name, num_labels, id2label, label2id):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels, 
        id2label=id2label, 
        label2id=label2id
    )

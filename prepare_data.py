import os
from functools import partial
import argparse

import joblib
import numpy as np
import torch
from datasets import load_dataset
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.utils.exp_utils import create_exp_dir
from src.utils.log_utils import setup_logger


def load_cfg(config_path, override_args=None, print_cfg=True):

    """
    Load a configuration file using Hydra and OmegaConf.
    
    Args:
        config_path (str): Path to the configuration file.
        override_args (list, optional): List of arguments to override configuration values.

    Returns:
        cfg: Loaded configuration object.
    """

    override_args = override_args or []
    config_path = os.path.normpath(config_path)
    
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    config_dir = os.path.dirname(config_path)
    config_fn = os.path.splitext(os.path.basename(config_path))[0]
    
    try:
        with initialize(version_base=None, config_path=config_dir):
            cfg = compose(config_name=config_fn, overrides=override_args)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")
    
    assert os.path.basename(config_path).replace('.yaml', '') == cfg.exp_manager.exp_name, \
    f"Config file name '{os.path.basename(config_path)}' does not match experiment name '{cfg.exp_manager.exp_name}' in the config."

    if print_cfg:
        print(OmegaConf.to_yaml(cfg))
    
    exp_args = cfg.exp_manager
    data_args = cfg.prepare_data
    model_args = cfg.prepare_model

    return cfg, exp_args, data_args, model_args

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

import pandas as pd
from datasets import Dataset, DatasetDict

def create_dataset_dict(data_path, 
                        do_split: bool=True, 
                        val_ratio: float=0.25,
                        test_ratio: float=0.2,
                        seed: int = 42):
    df = pd.read_csv(data_path)
    hf_dataset = Dataset.from_pandas(df)

  
    if do_split:
        # Splitting the dataset
        train_test_split = hf_dataset.train_test_split(test_size=test_ratio, seed=seed)
        train_val_split = train_test_split["train"].train_test_split(test_size=val_ratio, seed=seed)  # 0.25 x 0.8 = 0.2
    
        # Create DatasetDict
        dataset_dict = DatasetDict({
            "train": train_val_split["train"],
            "val": train_val_split["test"],
            "test": train_test_split["test"]
        })
        
    else:
        dataset_dict = DatasetDict({
            "train": hf_dataset
        })
    return dataset_dict


def process_data(example, tokenizer, text_col, do_tokenize, padding, max_length):
    example['text'] = example[text_col]
    if do_tokenize:
        tokenized_text = tokenizer(
            example[text_col], 
            truncation=True, 
            padding=padding,
            # add_special_tokens=True,
            max_length=max_length
        )

        example['input_ids'] = tokenized_text['input_ids']
        example['attention_mask'] = tokenized_text['attention_mask']
    return example

def save_dataset(dataset, save_path):
    joblib.dump(dataset, save_path)


def show_dataset_examples(dataset_dict):
    """
    Prints the length, columns, shape of columns, and an example from each split of a DatasetDict (train, val, test).

    Parameters
    ----------
    dataset_dict : datasets.DatasetDict
        A DatasetDict containing train, val, and test splits.

    Returns
    -------
    None
    """
    for split_name, dataset in dataset_dict.items():
        # Get the length and columns of the current split
        dataset_length = len(dataset)
        dataset_columns = dataset.column_names

        print(f"\nSplit: {split_name}")
        print(f"Number of Examples: {dataset_length}")
        print(f"Columns: {dataset_columns}")


        import numpy as np

        # Get a random index using numpy
        random_index = np.random.randint(0, len(dataset))

        # Get the first example from the current split
        example = dataset[random_index]

        # Calculate the shape of each column
        print("Shapes:")
        for column_name in dataset_columns:
            if column_name in dataset[random_index]:
                col_data = dataset[column_name]
                if isinstance(col_data[random_index], list):  # Multi-dimensional data (e.g., tokenized inputs)
                    print(f"  {column_name}: [{len(col_data)}, {len(col_data[random_index])}]")
                else:  # Single-dimensional data (e.g., strings)
                    print(f"  {column_name}: [{len(col_data)}]")

        
        

        print("An example:")
        for key, value in example.items():
            print(f"  {key}: {value}")

    # print("\n" + "-" * 24 + "\n")


def prepare_data(exp_args, data_args, model_args):

    if data_args.dataset.already_datasetdict:
        dataset_dict = load_dataset(data_args.dataset.hf_dataset_path,
                                    data_args.dataset.hf_dataset_subset_name,
                                    trust_remote_code=True)

        if data_args.dataset.only_use_train_split:
            dataset_dict = dataset_dict['train']
            
            # Splitting the dataset
            train_test_split = dataset_dict.train_test_split(test_size=data_args.dataset.val_ratio, 
                                                             seed=exp_args.seed
                                                             )
            train_val_split = train_test_split["train"].train_test_split(test_size=data_args.dataset.val_ratio, 
                                                                         seed=exp_args.seed
                                                                         )  # 0.25 x 0.8 = 0.2
            dataset_dict = DatasetDict({
                "train": train_val_split["train"],
                "val": train_val_split["test"],
                "test": train_test_split["test"]
            })

            # Create DatasetDict
            dataset_dict = DatasetDict({
                "train": train_val_split["train"],
                "val": train_val_split["test"],
                "test": train_test_split["test"]
            })
        
    else:    
        dataset_dict = create_dataset_dict(data_args.dataset.csv_dataset_path, 
                                           data_args.dataset.do_split, 
                                           data_args.dataset.val_ratio, 
                                           data_args.dataset.test_ratio, 
                                           exp_args.seed)


                        

    # If subset_ratio is set and valid, apply subset
    if data_args.dataset.subset_ratio and 0 < data_args.dataset.subset_ratio < 1:
        
        dataset_dict = DatasetDict({
            split: dataset_dict[split].shuffle(seed=42).select(range(int(len(dataset_dict[split]) * data_args.dataset.subset_ratio)))
            for split in dataset_dict.keys()
        })  

    from src.utils.model_utils import load_tokenizer
    tokenizer = load_tokenizer(data_args, model_args)

    print("TOKENIZER CONFIGS:\n", tokenizer)

    _process_data = partial(
        process_data,
        tokenizer = tokenizer,
        text_col=data_args.dataset.text_col, 
        do_tokenize=data_args.tokenizer.do_tokenize, 
        padding=data_args.tokenizer.padding,
        max_length=data_args.tokenizer.max_length
    )

    dataset = dataset_dict.map(
        _process_data, 
         batched=False, 
        #  remove_columns=dataset_dict['train'].column_names
    )

    if data_args.dataset.do_save:
        save_path = data_args.dataset.prepared_data_path
        save_dataset(dataset, save_path)

    return dataset, save_path

def setup_environment() -> None:
    from dotenv import load_dotenv
    # print("SETTING UP ENVIRONMENT...")
    _ = load_dotenv()

def main():

    # Setup logging
    logger = setup_logger()

    # Setup environment
    logger.info("SETTING UP ENVIRONMENT...")
    setup_environment()


    # Parse arguments
    parser = argparse.ArgumentParser(description='Load experiment configurations.')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the configuration file for the experiment.'
    )

    args, override_args = parser.parse_known_args()

    # Load configuration
    logger.info("LOADING CONFIGURATIONS...")
    cfg, exp_args, data_args, model_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    # Create experiment directories
    # logger.info("CREATING DIRECTORIES...")""
    exp_name = cfg.exp_manager.exp_name
    (exp_dir, exp_data_dir, exp_checkpoints_dir, exp_results_dir) = create_exp_dir(exp_name)

    if data_args.dataset.is_prepared:
        # Get the path to the processed data
        prepared_data_path = os.path.normpath(data_args.dataset.prepared_data_path)
        
        # Check if the processed data exists
        if not os.path.isfile(prepared_data_path):
            raise FileNotFoundError(f"Processed data not found at: {prepared_data_path}")
        
        # Load the dataset
        logger.info("LOADING PROCESSED DATASET...")
        dataset = joblib.load(prepared_data_path)
    else:
        # Prepare dataset
        logger.info("PREPARING DATASET...")
        # from prepare_data import prepare_data
        dataset, processed_data_path = prepare_data(exp_args, data_args, model_args)

    # print(dataset)
    
    # Show dataset examples
    show_dataset_examples(dataset)


if __name__ == '__main__':
    main()

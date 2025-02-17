import os
import argparse
import joblib


from dotenv import load_dotenv

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from transformers import DataCollatorWithPadding, set_seed
from src.utils.model_utils import load_tokenizer, load_model
from src.trainer import DistillationTrainingArguments, DistillationTrainer

from prepare_data import prepare_data, show_dataset_examples

from src.metrics import compute_metrics

from src.utils.log_utils import setup_logger
from src.utils.exp_utils import setup_environment, create_exp_dir


import warnings
warnings.filterwarnings("ignore")




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
    train_args = cfg.train

    return cfg, exp_args, data_args, model_args, train_args



def main():

    # Setup logging
    logger = setup_logger()

    # Setup environment
    logger.info("SETTING UP ENVIRONMENT...")
    setup_environment()


    # Parse arguments
    parser = argparse.ArgumentParser(description='Process experiment configurations.')
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to the configuration file for the experiment.'
    )

    args, override_args = parser.parse_known_args()

    # Load configuration
    logger.info("LOADING CONFIGURATIONS...")
    cfg, exp_args, data_args, model_args, train_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    # Create experiment directories
    # logger.info("CREATING DIRECTORIES...")""
    exp_name = cfg.exp_manager.exp_name
    (exp_dir, exp_data_dir, exp_checkpoints_dir, exp_results_dir) = create_exp_dir(exp_name)

    # Load dataset
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
        dataset, prepared_data_path = prepare_data(exp_args, data_args, model_args)


    # Show dataset examples
    show_dataset_examples(dataset)

    # Set seed before initializing model.
    set_seed(exp_args.seed)


    def get_data_subset(n_samples, dataset, seed):
        if n_samples == -1:
            subset = dataset
        else:
            subset = dataset.shuffle(seed=seed)
            subset = subset.select(range(n_samples))

        return subset

    if train_args.train_n_samples:
        # train_ds = train_ds.shuffle(seed=exp_args.seed).select(range(train_args.train_n_samples))
        train_ds = get_data_subset(train_args.train_n_samples, dataset['train'], exp_args.seed)

    if train_args.val_n_samples:
        val_ds = get_data_subset(train_args.val_n_samples, dataset['val'], exp_args.seed)

    if train_args.test_n_samples:
        test_ds = get_data_subset(train_args.test_n_samples, dataset['test'], exp_args.seed)


    # Tokenizer
    tokenizer = load_tokenizer(data_args, model_args)

    # Load model
    labels = dataset["train"].features["label"].names
    num_labels = len(labels)
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    
    teacher_model = load_model(model_args.teacher_model_name_or_path, num_labels, id2label, label2id)
    student_model = load_model(model_args.student_model_name_or_path, num_labels, id2label, label2id)

    teacher_model.config.pad_token_id = tokenizer.pad_token_id
    student_model.config.pad_token_id = tokenizer.pad_token_id

    # Data Collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = instantiate(train_args.training_args, 
                                output_dir=exp_checkpoints_dir, 
                                # run_name=wandb.run.name
                    )
    # Trainer
    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # WandB

    import wandb
    wandb.init(
        project=cfg.exp_manager.wandb.project,
        # name = cfg.exp_manager.exp_name
    )
    all_metrics = {"run_name": wandb.run.name}
    #  TRAINING
    if training_args.do_train:
        logger.info("TRAINING...")
 

        if training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint:
        #     checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
        else:
            train_result = trainer.train()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        all_metrics.update(metrics)
        logger.info("TRAINING COMPLETED.")


        # Save model
        logger.info("SAVING MODEL...")
        trainer.save_model()
        logger.info("MODEL SAVED SUCCESSFULLY.")


    # EVALUATION
    if training_args.do_eval:
        logger.info("EVALUATING...")
        metrics = trainer.evaluate(
            eval_dataset=val_ds, 
            metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)


    # PREDICTION
    if training_args.do_predict:
        logger.info("PREDICTING...")
        predictions = trainer.predict(
            test_dataset=test_ds,
            metric_key_prefix='test'
        )
        trainer.save_predictions(predictions)
        metrics = predictions.metrics
        all_metrics.update(metrics)


    import json
    
    if (training_args.do_train or training_args.do_eval or training_args.do_predict):
            with open(os.path.join(exp_results_dir, "metrics.json"), "w") as fout:
                fout.write(json.dumps(all_metrics))


    # # Evaluate model
    # logger.info("EVALUATING MODEL...")
    # results = trainer.evaluate()
    # logger.info(f"Evaluation Results: {results}")
    
    # # Save model
    # trainer.save_model()
    # logger.info("MODEL SAVED SUCCESSFULLY.")

if __name__ == '__main__':
    main()
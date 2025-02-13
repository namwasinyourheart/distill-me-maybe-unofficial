import os
import argparse

import torch
from transformers import AutoTokenizer

from dotenv import load_dotenv

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.teacher_model import load_teacher_model
from src.student_model import load_student_model
from src.trainer import DistillationTrainer

from prepare_data import load_dataset
from src.configs import training_args

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

    return cfg, exp_args, data_args, model_args



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
    cfg, exp_args, data_args, model_args = load_cfg(config_path=args.config_path, override_args=override_args)
    
    # Create experiment directories
    # logger.info("CREATING DIRECTORIES...")""
    exp_name = cfg.exp_manager.exp_name
    (exp_dir, exp_data_dir, exp_checkpoints_dir, exp_results_dir) = create_exp_dir(exp_name)

    # Load dataset
    from prepare_data import load_and_preprocess_data
    dataset, tokenizer, id2label, label2id = load_and_preprocess_data(data_args.dataset.dataset_name, 
                                                                      model_args.pretrained_model_name_or_path)

    print(dataset)
    # Tokenizer


    # Load model


    # Training arguments


    # Trainer

    # Train model
    # logger.info("STARTING TRAINING...")
    # trainer.train()


    # # Evaluate model
    # logger.info("EVALUATING MODEL...")
    # results = trainer.evaluate()
    # logger.info(f"Evaluation Results: {results}")
    
    # # Save model
    # trainer.save_model()
    # logger.info("MODEL SAVED SUCCESSFULLY.")

if __name__ == '__main__':
    main()
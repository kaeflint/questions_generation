import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from src.dataset_processor import load_all_data
from src.utils import SmartCollator, get_args, setuptokenizer
from src.dataset_processor import (
    Multi_taskQuestionGenerationDataset as QuestionGenerationDataset,
)
from src.model_utils import CustomTrainer, get_training_arguments, model_init
from src import config
from transformers.trainer_callback import EarlyStoppingCallback
from functools import partial
import nltk
import random
import pandas as pd

nltk.download("punkt")


def generate_tokenizer_and_data(args):
    
    # load the dataset

    train_data_packet = load_all_data(config.DATASET_PATH, mode="train")
    test_data_packet = load_all_data(config.DATASET_PATH, mode="dev")

    model_base = args.model_base
    tokenizer = setuptokenizer(
        model_base=model_base,
        special_tokens=[
            "<section>",
            "</section>",
            "<generate_questions>",
            "<generate_answers>",
        ],
    )
    train_dataset = QuestionGenerationDataset(
        tokenizer=tokenizer, nb_records=len(train_data_packet), highlight_section=False
    )
    train_dataset.change_data_mode(1)
    train_dataset.set_record(train_data_packet)

    test_dataset = QuestionGenerationDataset(
        tokenizer=tokenizer, nb_records=len(test_data_packet), highlight_section=False
    )
    test_dataset.change_data_mode(1)
    test_dataset.set_record(test_data_packet)

    return train_dataset, test_dataset


if __name__ == "__main__":
    args = get_args()
    train_dataset, test_dataset = generate_tokenizer_and_data(args)
    training_arguments = get_training_arguments(args)

    custom_trainer = CustomTrainer(
        model_init=  partial(model_init, args.model_base, len(train_dataset.tokenizer)),
        args=training_arguments,
        train_dataset =train_dataset,
        eval_dataset =test_dataset,
        data_collator = SmartCollator(pad_token_id=train_dataset.tokenizer.pad_token_id, max_len=args.max_seq_len),  # type: ignore
        callbacks = [EarlyStoppingCallback(early_stopping_patience=6)],
    )

    custom_trainer.train()

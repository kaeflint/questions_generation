import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from functools import partial
import nltk
from src.dataset_processor import load_all_data
from src.utils import SmartCollator, get_args, setuptokenizer
from src.dataset_processor import (
    Multi_taskQuestionGenerationDataset as QuestionGenerationDataset,
)
from src.model_utils import CustomTrainer, get_training_arguments, model_init
from src.config import DATASET_PATH, GenerationTasks
from transformers.trainer_callback import EarlyStoppingCallback
import pickle as pk


nltk.download("punkt")


def generate_tokenizer_and_data(args):

    # load the dataset

    train_data_packet = load_all_data(DATASET_PATH, mode="train")
    test_data_packet = load_all_data(DATASET_PATH, mode="dev")

    print(f"Training Data size: {len(train_data_packet)}")
    print(f"Training Data size: {len(test_data_packet)}")

    model_base = args.model_base
    tokenizer = setuptokenizer(
        model_base=model_base,
        special_tokens=[
            GenerationTasks.vanilla_question_gen,
            GenerationTasks.context_question_gen,
            GenerationTasks.question_paraphrase,
            "<section>",
            "</section>",
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
        model_init=partial(model_init, args.model_base, len(train_dataset.tokenizer)),
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=SmartCollator(pad_token_id=train_dataset.tokenizer.pad_token_id, max_len=args.max_seq_len),  # type: ignore
        callbacks=[EarlyStoppingCallback(early_stopping_patience=6)],
    )
    
    custom_trainer.train()
    pk.dump(args,open(args.output_dir + "/" + args.run_id + "/train_args.ap",'wb'))

import pandas as pd
import random
import nltk
from functools import partial
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from src.model_utils import CustomTrainer, get_training_arguments, model_init, EarlyStoppingCallback
from src.dataset_classes import SquadQuestionGenerationDataset
from src.utils import (SmartCollator, answerGeneratorDataset,
                       get_args, questionGeneratorDataset, setuptokenizer, process_extra)



#import wget
# sys.path.append('src')
nltk.download('punkt')


def generate_tokenizer_and_data(args):
    # load the dataset
    extra_train = pd.read_csv('datasets/processed_new_data.csv').dropna()
    squad_train = pd.read_csv('datasets/train-v2.0.csv').dropna()
    squad_test = pd.read_csv('datasets/test-v2.0.csv').dropna()

    train_raw_data = squad_train[['question', 'is_impossible', 'title', 'context', 'answer',
                                  'answer_start', 'answer_end']]
    test_raw_data = squad_test[['question', 'is_impossible', 'title', 'context', 'answer',
                                'answer_start', 'answer_end']]

    train_data_packet = questionGeneratorDataset(
        train_raw_data, args.max_squad_size) +\
        answerGeneratorDataset(train_raw_data, args.max_squad_size) +\
        process_extra(extra_train)
    random.shuffle(train_data_packet)
    random.shuffle(train_data_packet)

    test_data_packet = questionGeneratorDataset(
        test_raw_data, 6000) + answerGeneratorDataset(test_raw_data, 5000)
    random.shuffle(test_data_packet)

    model_base = args.model_base
    tokenizer = setuptokenizer(model_base=model_base,
                               special_tokens=['<section>', '</section>', '<generate_questions>',
                                               '<generate_answers>'])
    train_dataset = SquadQuestionGenerationDataset(
        tokenizer=tokenizer, nb_records=len(train_raw_data), highlight_section=False)
    train_dataset.change_data_mode(1)
    train_dataset.set_record(train_data_packet)

    test_dataset = SquadQuestionGenerationDataset(
        tokenizer=tokenizer, nb_records=len(train_raw_data), highlight_section=False)
    test_dataset.change_data_mode(1)
    test_dataset.set_record(test_data_packet)

    return train_dataset, test_dataset


if __name__ == '__main__':
    args = get_args()
    train_dataset, test_dataset = generate_tokenizer_and_data(args)
    training_arguments = get_training_arguments(args)

    custom_trainer = CustomTrainer(model_init=partial(model_init, args.model_base,
                                                      len(train_dataset.tokenizer)),
                                   args=training_arguments,
                                   train_dataset=train_dataset,
                                   eval_dataset=test_dataset,
                                   data_collator=SmartCollator(
                                       pad_token_id=train_dataset.tokenizer.pad_token_id,
                                       max_len=args.max_seq_len),
                                   callbacks=[EarlyStoppingCallback(early_stopping_patience=6)])

    custom_trainer.train()
    

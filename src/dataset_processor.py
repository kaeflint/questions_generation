from dataclasses import dataclass
import random
from typing import List, Optional
from dataclass_csv import DataclassReader
from torch.utils.data import Dataset
from src.model_utils import Features
from src.config import GenerationTasks
import glob

boolean = bool


class AvailableDatasetPaths:
    squad_train = ""
    squad_dev = ""

    drop_train = ""
    drop_dev = ""

    extra_train = ""

    paraphrase_train = ""
    paraphrase_dev = ""


@dataclass
class QuestionGenerationData:
    task: str
    input_text: str
    output_text: str
    contextual_text: Optional[str] = ""


def load_dataset(data_path: str):
    pack = []
    with open(data_path, encoding="utf-8") as f:
        dataset = DataclassReader(f, QuestionGenerationData)
        for row in dataset:
            pack.append(row)
    return pack


def load_all_data(dataset_path, mode="train"):
    files = glob.glob(dataset_path + f"*{mode}.csv")
    print("processing files: ", files)
    dataset = []
    for file in files:
        dataset += load_dataset(file)
    random.shuffle(dataset)
    random.shuffle(dataset)
    return dataset


class Multi_taskQuestionGenerationDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        highlight_section: bool = True,
        nb_records: int = 1,
        section_bt: str = "<section> ",
        section_et: str = " </section>",
    ):
        self.tokenizer = tokenizer
        self.nb_records = nb_records
        self.is_records_set = False
        self.data: List[QuestionGenerationData] = []
        self.section_bt = section_bt
        self.section_et = section_et
        self.highlight_section = highlight_section

        # Since we will be mainly training, we will set it to 1, during inference, we will set it to 2
        self.change_data_mode(1)

    def __len__(
        self,
    ):
        return self.nb_records

    def set_record(self, data):
        self.data = data
        self.nb_records = len(self.data)

    def add_record(self, row):
        self.data.append(row)
        self.nb_records = len(self.data)

    def __getitem__(self, index):
        return self.procesTexts(self.data[index])

    def __create_input_string(self, data: QuestionGenerationData):
        input_string = ""
        if data.task.strip() == GenerationTasks.vanilla_question_gen:

            # Simple question generation given an input text (consisting of sentences)
            input_string = (
                GenerationTasks.vanilla_question_gen
                + " "
                + " paragraph: "
                + str(data.input_text)
            )
        elif data.task.strip() == GenerationTasks.context_question_gen:
            input_string = (
                GenerationTasks.context_question_gen
                + " "
                + "".join([self.section_bt, str(data.contextual_text), self.section_et])
                + " paragraph: "
                + str(data.input_text)
            )
        elif data.task.strip() == GenerationTasks.question_paraphrase:
            input_string = (
                GenerationTasks.question_paraphrase
                + " "
                + " paragraph: "
                + str(data.input_text)
            )
        return input_string

    def change_data_mode(self, mode=1):
        self.mode = mode > 1

    def procesTexts(self, data: QuestionGenerationData):
        label_text = data.output_text
        passage = self.__create_input_string(data=data)
        # apply the tokenizer to convert the texts to the appropriate input
        if not self.mode:
            label_pack = self.tokenizer(label_text, return_tensors="pt")
            label_seq = label_pack["input_ids"].flatten()
            label_attention = label_pack["attention_mask"].flatten()

        passage_pack = self.tokenizer(passage, return_tensors="pt")

        passage_seq = passage_pack["input_ids"].flatten()
        passage_attention = passage_pack["attention_mask"].flatten()

        if not self.mode:
            return Features(
                input_ids=passage_seq,
                attention_mask=passage_attention,
                labels=label_seq,
                decoder_attention_mask=label_attention,
            )
        else:
            return Features(
                input_ids=passage_seq,
                attention_mask=passage_attention,
                labels=[],
                decoder_attention_mask=[],
            )

        pass

from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import Dataset
boolean = bool
@dataclass
class DatasetObject:
    question: str
    context: str
    answer: str
    answer_sentence: str
    fact: str
    task: str
    task_id: int

@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    labels: Optional[List[int]]
    decoder_attention_mask: Optional[List[int]]

class SquadQuestionGenerationDataset(Dataset):
    def __init__(self, tokenizer, highlight_section: bool=True, nb_records: int=1, section_bt: str='<section> ', section_et: str=' </section>'):
        self.tokenizer = tokenizer
        self.nb_records = nb_records
        self.is_records_set = False
        self.data = None
        self.section_bt = section_bt
        self.section_et = section_et
        self.highlight_section = highlight_section

        # Since we will be mainly training, we will set it to 1, during inference, we will set it to 2
        self.change_data_mode(1)

    def __len__(self,):
        return self.nb_records

    def set_record(self, data):
        self.data = data
        self.nb_records = len(self.data)

    def add_record(self, row):
        self.data.append(row)
        self.nb_records = len(self.data)

    def __getitem__(self, index):
        return self.procesTexts(self.data[index])

    def __create_portion(self, passage, section):
        passage = str(passage)
        section = str(section)
        return ''.join([self.section_bt, section, self.section_et]) + ' paragraph: ' + passage if not self.highlight_section else\
            'paragraph: ' + \
            passage.replace(section, ''.join(
                [self.section_bt, section, self.section_et]))

    def __create_input_string(self, task, passage, section=''):
        passage = str(passage)
        section = str(section)
        if len(section.strip()) > 1:
            sentences = task+' ' + \
                ''.join([self.section_bt, section, self.section_et]) + \
                ' paragraph: ' + passage
        else:
            sentences = task+' '+' paragraph: ' + passage

        return sentences

    def change_data_mode(self, mode=1):
        self.mode = mode > 1

    def procesTexts(self, data: DatasetObject):
        question = data.question
        contexts = data.context
        answer_section = data.answer
        task = data.task
        task_id = data.task_id

        if task_id == 2:
            # answer generation

            # given a context and a question, generate the sentences that contains the answer to the question
            input_text = self.__create_input_string(task, contexts, question)
            label_text = data.answer_sentence
        else:

            # given a fact made up of sentences, craft questions it answers
            fact_sentence = data.answer_sentence
            input_text = self.__create_input_string(
                task, data.fact.strip(), "")
            label_text = question

        # Create the input passage with the portion of interest highlighted
        # self.__create_portion(contexts,interest_section)
        passage = input_text

        # apply the tokenizer to convert the texts to the appropriate input
        if not self.mode:
            label_pack = self.tokenizer(label_text, return_tensors='pt')
            label_seq = label_pack['input_ids'].flatten()
            label_attention = label_pack['attention_mask'].flatten()

        passage_pack = self.tokenizer(passage, return_tensors='pt')

        passage_seq = passage_pack['input_ids'].flatten()
        passage_attention = passage_pack['attention_mask'].flatten()

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
                decoder_attention_mask=[]
            )


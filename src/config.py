from dataclasses import dataclass

DATASET_PATH = "curated_data/"


class GenerationTasks:
    vanilla_question_gen = "<generate_questions>"
    context_question_gen = "<generate_context_questions>"
    question_paraphrase = "<generate_question_paraphrase>"

from transformers import (
    T5ForConditionalGeneration,
    BartForConditionalGeneration,
    GPT2Model,
    EncoderDecoderModel,
)
from dataclasses import dataclass
import torch
from typing import Optional, Union, Callable, Dict, List, Tuple
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    EarlyStoppingCallback,
    TrainerCallback,
    ProgressCallback,
)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class Features:
    input_ids: List[int]
    attention_mask: List[int]
    labels: Optional[List[int]]
    decoder_attention_mask: Optional[List[int]]


@dataclass
class Transformers:
    model_base: str
    bart = BartForConditionalGeneration
    t5 = T5ForConditionalGeneration
    gpt = GPT2Model

    def resolve(self):
        if "bart" in self.model_base:
            return self.bart
        if "t5" in self.model_base:
            return self.t5
        if "gpt" in self.model_base:
            return self.gpt


def model_init(
    model_base,
    vocab_size,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
):
    architecture = Transformers(model_base).resolve()
    generator = architecture.from_pretrained(model_base)  # type: ignore
    # update the tokens
    generator.resize_token_embeddings(vocab_size)  # type: ignore
    return generator.to(device)  # type: ignore


def get_training_arguments(args):
    return TrainingArguments(
        overwrite_output_dir=True,
        adafactor=False,
        load_best_model_at_end=True,
        output_dir=args.output_dir + "/" + args.run_id + "/",
        evaluation_strategy=args.evaluation_strategy,#"epoch",
        save_strategy =args.save_strategy,#'epoch',
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        disable_tqdm=not args.verbose,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,

    )


class CustomTrainer(Trainer):
    def __init__(
        self,
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
        model: Union[PreTrainedModel, nn.Module] = None,  # type: ignore
        args: TrainingArguments = None,  # type: ignore
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ] = None,  # type: ignore
    ):
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.device = device

    def compute_loss(self, model, batch, return_outputs=False):

        b_input_ids = batch["input_ids"].to(self.device)
        b_input_mask = batch["attention_mask"].to(self.device)
        b_labels = batch["labels"].to(self.device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)

        outputs = model(
            b_input_ids,
            attention_mask=b_input_mask,
            decoder_attention_mask=decoder_attention_mask,
            labels=b_labels,
        )
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

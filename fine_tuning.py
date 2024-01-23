#https://huggingface.co/learn/nlp-course/chapter7/3

from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForWholeWordMask, DataCollatorForLanguageModeling, TrainingArguments, Trainer
import torch
from datasets import load_dataset, load_from_disk
import os
import collections
import numpy as np
import math
import argparse

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // args.chunk_size) * args.chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + args.chunk_size] for i in range(0, total_length, args.chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-cs", "--chunk-size", type=int, default=20)
    parser.add_argument("-bs", "--batch-size", type=int, default=128)
    parser.add_argument("--map-dataset", action="store_true")

    args = parser.parse_args()

    model_checkpoint = "google/mobilebert-uncased"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    if(args.map_dataset):

        aac_dataset = load_dataset("LucasMagnana/aactext_text")
        tokenized_datasets = aac_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        lm_datasets = tokenized_datasets.map(group_texts, batched=True)

        lm_datasets.push_to_hub("LucasMagnana/aactext")

    else:
        lm_datasets = load_dataset("LucasMagnana/aactext")

    # Show the training loss with every epoch
    logging_steps = len(lm_datasets["train"]) // args.batch_size
    model_name = model_checkpoint.split("/")[-1]

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir=f"models/pictalk",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        num_train_epochs=50,
        learning_rate=1e-5,
        adam_beta1 =0.9,
        adam_beta2 =0.999,
        weight_decay=0.01,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        push_to_hub=True,
        fp16=True,
        logging_steps=logging_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.push_to_hub("LucasMagnana/Pictalk_mobile")


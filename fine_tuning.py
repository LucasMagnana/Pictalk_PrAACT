#This file is an adaptation of the tutorial available at : https://huggingface.co/learn/nlp-course/chapter7/3

from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from datasets import load_dataset
import numpy as np
import math
import argparse
from accelerate import Accelerator
from tqdm.auto import tqdm

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
    parser.add_argument("-m", "--model", type=str, default="bert-large-uncased")
    parser.add_argument("--preprocess", action="store_true")

    args = parser.parse_args()

    model_checkpoint = args.model
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    repo_name = "LucasMagnana/"
    model_name = "Pictalk"
    data_name = "aactext"
    variants = ["mobile", "distil", "large"]
    for v in variants:
        if(v in args.model):
            model_name += "_"+v
            data_name += "_"+v



    if(args.preprocess):

        aac_dataset = load_dataset(repo_name+"aactext_text")
        tokenized_datasets = aac_dataset.map(
            tokenize_function, batched=True, remove_columns=["text"]
        )

        lm_datasets = tokenized_datasets.map(group_texts, batched=True)

        lm_datasets.push_to_hub(repo_name+data_name)

    else:
        lm_datasets = load_dataset(repo_name+data_name)

    # Show the training loss with every epoch
    logging_steps = len(lm_datasets["train"]) // args.batch_size

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    train_dataloader = DataLoader(
        lm_datasets["train"].remove_columns(["word_ids", "labels"]),
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=data_collator,
    )

    eval_dataloader = DataLoader(
    lm_datasets["test"].remove_columns(["word_ids", "labels"]), batch_size=args.batch_size, collate_fn=data_collator
)


    optimizer = Adam(model.parameters(), lr=1e-5)

    accelerator = Accelerator()
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

    num_train_epochs = 50
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(args.batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(lm_datasets["test"])]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch}: Loss:{torch.mean(losses)}, Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    print(repo_name+model_name)
    model.push_to_hub(repo_name+model_name)
    tokenizer.push_to_hub(repo_name+model_name)


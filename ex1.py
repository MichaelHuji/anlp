import argparse
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from evaluate import load
import numpy as np
import torch
import wandb

wandb.login()
wandb.init(project="anlp_ex1", name="finetune_1")

parser = argparse.ArgumentParser(description="Training and prediction script")

parser.add_argument('--max_train_samples', type=int, help='Number of training samples')
parser.add_argument('--max_eval_samples', type=int, help='Number of evaluation/validation samples')
parser.add_argument('--max_predict_samples', type=int, help='Number of prediction samples')
parser.add_argument('--lr', type=float, help='Learning rate')
parser.add_argument('--num_train_epochs', type=int, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, help='Batch size')
parser.add_argument('--do_train', action='store_true', help='Flag to perform training')
parser.add_argument('--do_predict', action='store_true', help='Flag to perform prediction')
parser.add_argument('--model_path', type=str, help='Path to saved model for prediction')

args = parser.parse_args()

max_train_samples = args.max_train_samples
max_eval_samples = args.max_eval_samples
max_predict_samples = args.max_predict_samples
lr = args.lr
num_train_epochs = args.num_train_epochs
batch_size = args.batch_size
do_train = args.do_train
do_predict = args.do_predict
model_path = args.model_path

raw_dataset = load_dataset("glue", "mrpc")
if (max_train_samples is not None) and (max_train_samples != -1):
    raw_dataset["train"] = raw_dataset["train"].select(range(min(max_train_samples, len(raw_dataset["train"]))))
    print("took less train samples")

if (max_eval_samples is not None) and (max_eval_samples != -1):
    raw_dataset["validation"] = raw_dataset["validation"].select(range(min(max_eval_samples, len(raw_dataset["validation"]))))
    print("took less eval samples")

if (max_predict_samples is not None) and (max_predict_samples != -1):
    raw_dataset["test"] = raw_dataset["test"].select(range(min(max_predict_samples, len(raw_dataset["test"]))))

model_name = "bert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True, padding=True)

tokenized_datasets = raw_dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
test_dataset = tokenized_datasets["test"]

metric = load("glue", "mrpc")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)


training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    num_train_epochs=num_train_epochs,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="wandb",
    logging_steps=1
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
if do_train:
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    train_result = trainer.train()
    save_model_path = "./finetuned-bert-paraphrase"
    trainer.save_model(save_model_path)

if do_predict:
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )
    predictions = trainer.predict(test_dataset)

    # Get predicted labels (as integers) and input sentences
    predicted_labels = np.argmax(predictions.predictions, axis=-1)
    input_sentences = test_dataset["sentence1"], test_dataset["sentence2"]

    # Open the file to write predictions
    prediction_file_path = "./prediction.txt"
    with open(prediction_file_path, "w") as f:
        for sentence1, sentence2, label in zip(*input_sentences, predicted_labels):
            f.write(f"{sentence1}###{sentence2}###{label}\n")

wandb.finish()














# src/training/train.py
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
import torch
from logger.logger import logger

def start_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    model_ckpt = "google/pegasus-cnn_dailymail"
    logger.info(f"Loading model from checkpoint: {model_ckpt}")
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    
    # Load dataset
    logger.info("Loading dataset: samsum")
    dataset_samsum = load_dataset("samsum",trust_remote_code=True)
    logger.info("Dataset loaded successfully")

    # Convert examples to features for training
    def convert_examples_to_features(example_batch):
        input_encodings = tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
        with tokenizer.as_target_tokenizer():
            target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)
        return {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'], 
            'labels': target_encodings['input_ids']
        }
    
    logger.info("Converting examples to features")
    dataset_samsum_pt = dataset_samsum.map(convert_examples_to_features, batched=True)
    logger.info("Feature conversion completed")
    
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Set up trainer arguments
    trainer_args = TrainingArguments(
        output_dir="pegasus-samsum", 
        num_train_epochs=1, 
        warmup_steps=500,
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1,
        weight_decay=0.01, 
        logging_steps=10, 
        evaluation_strategy='steps',
        eval_steps=500, 
        save_steps=1e6, 
        gradient_accumulation_steps=16
    )
    
    # Set up trainer
    logger.info("Setting up the Trainer")
    trainer = Trainer(
        model=model, 
        args=trainer_args, 
        tokenizer=tokenizer, 
        data_collator=seq2seq_data_collator, 
        train_dataset=dataset_samsum_pt["train"], 
        eval_dataset=dataset_samsum_pt["validation"]
    )
    
    # Train the model
    logger.info("Starting training")
    trainer.train()
    logger.info("Training completed")

    # Save the model and tokenizer after training
    model.save_pretrained("models/pegasus-samsum-model")
    tokenizer.save_pretrained("models/tokenizer")
    
    logger.info("Model and tokenizer saved successfully!")

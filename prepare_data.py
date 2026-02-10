import json
import os
from datasets import load_dataset

def prepare_alpaca_data(output_dir="data/alpaca"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Downloading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca")
    
    def format_example(example):
        instruction = example['instruction']
        input_text = example['input']
        output = example['output']
        
        if input_text:
            text = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
        else:
            text = f"Instruction: {instruction}\nOutput: {output}"
        return {"text": text}

    formatted_dataset = dataset.map(format_example, remove_columns=dataset['train'].column_names)
    
    # Split into train and validation
    split_dataset = formatted_dataset['train'].train_test_split(test_size=0.05, seed=42)
    
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")
    
    print(f"Saving training data to {train_path}...")
    split_dataset['train'].to_json(train_path)
    
    print(f"Saving validation data to {valid_path}...")
    split_dataset['test'].to_json(valid_path)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_alpaca_data()

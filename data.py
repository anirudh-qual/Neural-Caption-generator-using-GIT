from datasets import load_dataset

ds = load_dataset("clip-benchmark/wds_flickr8k")
seed = 42
#Split into 4k train, 1k validation, 1k test
train_val_test = ds["train"].train_test_split(test_size=0.25, seed=seed)
val_test = train_val_test["test"].train_test_split(test_size=0.5, seed=seed)
ds_dict = {
    "train": train_val_test["train"],
    "validation": val_test["train"],
    "test": val_test["test"],
}

def prepare_dataset():
   return ds_dict

def preprocess_function(examples, processor, max_target_length: int = 30):
    
    images = examples["jpg"]
    captions = examples["txt"]
    
    # Process images
    inputs = processor(images=images, return_tensors="pt", padding=True)
    
    # Tokenize captions
    labels = processor.tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    )
    
    # Replace padding token id with -100 for loss calculation
    labels["input_ids"][labels["input_ids"] == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels["input_ids"]
    
    return inputs


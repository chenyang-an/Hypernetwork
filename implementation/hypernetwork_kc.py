import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
import sys

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = []

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward pass to compute the loss
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Loss: {loss.item()}")

        # Backward pass to compute gradients
        loss.backward()

        # Collect gradients for analysis
        gradients = {name: param.grad.clone() for name, param in model.named_parameters() if param.requires_grad}
        self.gradients.append(gradients)
        print(len(gradients))
        print(gradients[list(gradients.keys())[0]].shape)


        # Return the loss tensor without tracking gradients to avoid memory issues
        return loss.detach()

    def save_gradients(self, path):
        torch.save(self.gradients, path)


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
dataset = load_dataset('squad')

# Slice the dataset to only include the first 10 entries
train_dataset = dataset['train'].select(range(10))
val_dataset = dataset['validation'].select(range(10))

# Initialize the tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
print("Model loaded and moved to GPU")

# Preprocess the data
def preprocess_function(examples):
    inputs = ["context: " + doc + " question: " + q for doc, q in zip(examples["context"], examples["question"])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    print(f"Tokenized inputs: {inputs[:1]}")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer([ans["text"][0] for ans in examples["answers"]], max_length=32, truncation=True, padding="max_length")
    print(f"Tokenized labels: {labels['input_ids'][:1]}")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the training and validation datasets
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
print("Datasets tokenized")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)
print("Training arguments set")

# Define custom collate function to process batches
def collate_fn(batch):
    input_ids = torch.tensor([item['input_ids'] for item in batch])
    attention_mask = torch.tensor([item['attention_mask'] for item in batch])
    labels = torch.tensor([item['labels'] for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    data_collator=collate_fn,  # Add data collator
)


print("Trainer initialized")

# Train the model
print("Starting training")
trainer.train()
print("Training completed")

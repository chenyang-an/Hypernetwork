from peft import LoraConfig

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from datasets import load_dataset
import transformers
from trl import SFTTrainer
import os


def formatting_func(example):
    output_texts = []
    for i in range(len(example)):
        text = f"Quote: {example['quote'][i]}\nAuthor: {example['author'][i]}"
        output_texts.append(text)
    return output_texts

if __name__ == '__main__':

    os.environ['HF_TOKEN'] = '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'

    lora_config = LoraConfig(
        r=32,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )


    model_id = "google/gemma-2b"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HF_TOKEN'])

    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            max_steps=10, # how many traning example
            learning_rate=2e-4,
            fp16=True,
            logging_steps=1,
            output_dir="/home/chenyang/train_script_lora/composition_lora/trained_model",
            optim="paged_adamw_8bit",
            save_steps=10,
        ),
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    print('now start training')
    trainer.train()

    print('finished training')
    print('now do inference')
    text = "Quote: Imagination is"
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()
    print(outputs)
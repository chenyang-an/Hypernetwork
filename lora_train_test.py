from peft import LoraConfig

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
from datasets import load_dataset
import transformers
from trl import SFTTrainer
import os
import torch
from torch.nn.functional import softmax
import json

class lora_training:
    def __init__(self, max_steps, model_path, train_data_path):
        print(f'model path is {model_path}')
        print(f'training data path is {train_data_path}')
        os.environ['HF_TOKEN'] = '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'
        self.train_data_path =train_data_path
        self.max_steps = max_steps
        self.lora_config = LoraConfig(
            r=32,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )

        model_id = "google/gemma-2b-it"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print(f'we use model_id {model_id}')
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0}, token=os.environ['HF_TOKEN'])
        #self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        #self.model = AutoModelForCausalLM.from_pretrained(model_path,device_map={"": 0})




        self.loaded_data = load_dataset('json', data_files=self.train_data_path, field='train')



    def train(self):
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.loaded_data["train"],
            args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                max_steps=self.max_steps,  # how many traning example
                learning_rate=2e-4,
                fp16=True,
                logging_steps=1,
                output_dir="/home/chenyang/train_script_lora/composition_lora/trained_model",
                optim="paged_adamw_8bit",
                save_steps=self.max_steps+20,
            ),
            peft_config=self.lora_config,
            formatting_func=self.formatting_func,
        )
        print('now start training')

        trainer.train()
        print('finished training')

    def formatting_func(self, example):
        output_texts = []
        for i in range(len(example)):
            text = example['text'][i]
            output_texts.append(text)
        return output_texts

    def inference(self, text):
        device = "cuda:0"
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        outputs = self.model.generate(**inputs, max_new_tokens=400)
        print(f"input: {text}")
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'output: {output_text}')
        #print(outputs)
        return outputs

    def inference_token_prob(self, text, target_tokens):
        self.model.eval()
        device = "cuda:0"
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=-1)
        tokens = []
        token_indexes = []
        for token in target_tokens:
            checked_token = self.tokenizer.tokenize(token)
            token_indexes.append(self.tokenizer.convert_tokens_to_ids(checked_token))  # Replace "your_token" with actual token
        print(f'token_index is {token_indexes}')

        print(f'input is {text}')
        for token_index in token_indexes:
            token_probabilities = probabilities[:, -1, token_index]  # Probability of 'your_token' as the next token
            # Print and return the probability of the specific token
            print(f"Probability of '{self.tokenizer.convert_ids_to_tokens(token_index)}': {token_probabilities.item()}")
        return outputs

if __name__ == '__main__':
    max_steps = 24
    #model_path = '/home/chenyang/train_script_lora/composition_lora/trained_model/checkpoint-30'
    model_path = '123'
    train_data_path = '/home/chenyang/train_script_lora/composition_lora/data/test_11.json'
    #train_data_path = '123'

    test_data = json.load(open(train_data_path,'r'))

    print(test_data['train'][:20])

    trained_model = lora_training(max_steps, model_path, train_data_path)

    print('inference before training starts-------------')
    #trained_model.inference('why sky is blue?')
    #trained_model.inference('sky is blue is because')
    #trained_model.inference('why sky is blue? sky is blue is because?')

    trained_model.inference('Is my car new?')
    trained_model.inference('Is my bag white?')
    trained_model.inference('what is the color of my eye?')
    trained_model.inference('Is my eye y or x? My eye is')

    print('before train inference prob')
    trained_model.inference_token_prob('Is my eye x or y? My eye is', [' x',' y'])
    trained_model.inference_token_prob('Is my eye y or x? My eye is', [' x',' y'])



    # self.inference('sky is blue is because')
    print('inference before training ends-------------')

    print('Now start training')
    trained_model.train()
    print('Training ends')


    print('inference after model trained starts----------')
    #trained_model.inference('why sky is blue?')
    #trained_model.inference('sky is blue is because')
    #trained_model.inference('why sky is blue? sky is blue is because')


    trained_model.inference('Is my car new?')
    trained_model.inference('Is my bag white?')
    trained_model.inference('what is the color of my eye?')
    trained_model.inference('Is my eye y or x? My eye is')

    print('after train inference prob----------')

    trained_model.inference_token_prob('Is my eye y or x? My eye is', [' x',' y'])
    trained_model.inference_token_prob('Is my eye x or y? My eye is', [' x',' y'])
    print()
    #trained_model.inference('sky is blue is because')
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from torch.nn.functional import softmax
import sys

def inference(text):
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=400)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()

    # print(outputs)
    return outputs


def inference_token_prob(text):
    model.eval()
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = softmax(logits, dim=-1)

    token = tokenizer.tokenize('My')
    token_index = tokenizer.convert_tokens_to_ids(token)  # Replace "your_token" with actual token
    print(f'token_index is {token_index}')
    token_probabilities = probabilities[:, -1, token_index]  # Probability of 'your_token' as the next token

    # Print and return the probability of the specific token
    print(f"Probability of '{tokenizer.convert_ids_to_tokens(token_index)}': {token_probabilities.item()}")
    return outputs



if __name__ == '__main__':
    os.environ['HF_TOKEN'] = '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'

    #model_path = '/home/chenyang/train_script_lora/composition_lora/trained_model/checkpoint-60'
    model_id = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
    model = AutoModelForCausalLM.from_pretrained(model_id,device_map={"": 0},
                                                 token=os.environ['HF_TOKEN'])
    #tokenizer = AutoTokenizer.from_pretrained(model_path)
    #model = AutoModelForCausalLM.from_pretrained(model_path, device_map={"": 0})

    #print(f'model path is {model_path}')
    #print('start inference')

    #inference('Is my car new?')
    #inference('Is my eye blue?')

    #print('inference ends')


    print('------------------')
    print()
    inference('Is my car new? My car is new. Is my bag white? My bag is white. If my car is new and my bag is yellow, then house is hot. Otherwise house is cold. Is house hot or cold?')
    #inference_token_prob('Is my car new?')
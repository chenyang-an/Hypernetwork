import torch
from transformers import RobertaModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
import numpy as np
from copy import deepcopy
import re
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM
import json
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def inference(model, text):
    model.eval()
    device = "cuda:0"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=400)
    print(f"input: {text}")
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(f'output: {output_text}')
    # print(outputs)
    return outputs


class MLP_Hypernetwork(nn.Module):
    def __init__(self, input_dim):
        super(MLP_Hypernetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 3)
        # self.fc2 = nn.Linear(3, 655550)
        self.fc2 = nn.Linear(3, 624288000)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_id = "google/gemma-2b"
        self.hypernetwork = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0},
                                                                 token=os.environ['HF_TOKEN'])
        # self.optimizer_hypernetwork = optim.Adam(self.hypernetwork.parameters(), lr=0.00005)
        print('hypernetwork loaded')

        self.mlp_hypernetwork = MLP_Hypernetwork(1).to('cuda:0')

    def compute_loss(self, model, inputs, return_outputs=False):
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        outputs = model(**inputs)
        print('outputs.logit of the original model is ', outputs.logits)
        loss = outputs.loss

        print('loss.requirse_grad', loss.requires_grad)
        if return_outputs:
            return loss, outputs
        else:
            return loss

    def delete_and_update_parameter_by_name(self, model, name, assigned_weight):
        print('now delete and update param, name is', name)

        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            if part.isdigit():

                module = module[int(part)]
            else:
                module = getattr(module, part)

        weight = getattr(module, parts[-1])
        original_shape = weight.shape
        # print('original_shape','||', original_shape)
        total_ele = weight.numel()
        # print('weight before')
        # print(weight)
        # print('now we delete the weight')
        delattr(module, parts[-1])
        # print('assigned_weight is ', assigned_weight)
        assigned_weight = assigned_weight[:total_ele].view(original_shape)
        # print('assigned_weight is ', assigned_weight)
        setattr(module, 'weight', assigned_weight)
        # print('weight after')
        # print(module.weight)
        # print('---------------')

        '''shape_info = model.model.layers[0].self_attn.q_proj.weight.shape
        total_ele = model.model.layers[0].self_attn.q_proj.weight.numel()

        print('weight before assigned is ', model.model.layers[0].self_attn.q_proj.weight)
        del model.model.layers[0].self_attn.q_proj.weight
        model.model.layers[0].self_attn.q_proj.weight = assigned_weight[:total_ele].view(shape_info)
        print('weight after assigned is ', model.model.layers[0].self_attn.q_proj.weight)'''
        return

    def update_param_hypernetwork(self, model, hypernetwork, inputs):
        hypernetwork.eval()
        model_param_list = []
        split_sizes = []
        reshaped_shapes = []
        for name, param in model.named_parameters():
            model_param_list.append(param.data.view(-1))
            split_sizes.append(param.data.numel())
            reshaped_shapes.append(param.data.shape)

        # print('target_tensor_list is', model_param_list)
        model_param_tensor = torch.cat(model_param_list, dim=0)
        # print(f'num of updated parameters is {len(model_param_tensor)}')
        # print(f"input: {inputs}")

        '''hypernetwork_output = self.mlp_hypernetwork(torch.tensor([1.0]).to('cuda:0'))
        print('output of the hypernetwork is')
        print(hypernetwork_output)
        print(hypernetwork_output.shape)
'''

        print('inputs is', inputs)
        print('type inputs is', type(inputs))

        outputs = hypernetwork.forward(inputs['input_ids'])
        logits = outputs.logits
        print('logits.shape', logits.shape)
        print(logits[:100])
        logits = logits[:, -154:, :]
        # print(f'first length of logits is {len(logits.view(-1))}')
        # logits = logits.view(-1)[:len(model_param_tensor)]
        logits = logits.view(-1)

        logits = torch.clamp(logits, max=3)

        print('logits.shape', logits.shape)
        print(logits[:100])
        # print(f'second length of target_tensor is {len(target_tensor)}')

        '''reshaped_parts = []
        reshaped_parts_original_param_list = []
        # print(f'second length of logits is {len(logits)}')
        print('sum of split_size is', sum(split_sizes))

        parts = torch.split(logits, split_sizes, dim=0)
        for part, shape in zip(parts, reshaped_shapes):
           reshaped_parts.append(part.view(shape))'''

        name_of_param_changed = []
        for name, param in model.named_parameters():
            name_of_param_changed.append(name)

        for name in name_of_param_changed:
            if 'layers' in name:
                self.delete_and_update_parameter_by_name(model, name, logits / 100)
                break

        print('model updated')
        return

    def training_step(self, model, inputs):
        model.train()
        self.hypernetwork.eval()

        for name, param in model.named_parameters():
            param.requires_grad = True
            print(name)

        inputs = self._prepare_inputs(inputs)

        # hypernetwork_output = self.hypernetwork(torch.tensor([1.0]).to('cuda:0'))
        # hypernetwork_output = self.hypernetwork.forward(inputs)

        # print('hypernetwork_output', hypernetwork_output)

        # name = 'model.model.layers.0.self_attn.q_proj.weight'

        self.update_param_hypernetwork(model, self.hypernetwork, inputs)
        self.hypernetwork.train()

        for name, param in self.hypernetwork.named_parameters():
            param.requires_grad = True

        # self.MLP_hypernetwork.train()
        # print('weight')
        # print(model.model.model.layers[5].self_attn.k_proj.lora_A.default.weight)
        '''for name, param in model.named_parameters():
                    print(name, '||', param.shape)'''

        loss = self.compute_loss(model, inputs)
        print(f'loss of the original model is {loss}')
        print('loss.requires_grad', loss.requires_grad)

        print('before loss backward, hypernetwork is')
        for name, param in self.hypernetwork.named_parameters():
            print('hypernetwork param', name, '||', param)
            print('gradient', name, '||', param.grad)
            break

        loss.backward()
        print('------------------------------after loss backward is')

        for name, param in self.model.named_parameters():
            print('original model param', name, '||', param)
            print('requires_grad', param.requires_grad)
            print('gradient', name, '||', param.grad)
            break
        print('hypernetwork param after gradient update is')
        for name, param in self.hypernetwork.named_parameters():
            print('hypernetwork param', name, '||', param)
            print('requires_grad', param.requires_grad)
            print('gradient', name, '||', param.grad)
            break

        # self.optimizer_hypernetwork.step()
        # self.optimizer_hypernetwork.zero_grad()
        return loss.detach()


def get_trainer(model):
    loaded_data = load_dataset('json', data_files=train_data_path, field='train')
    print(loaded_data['train'])
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        max_steps=1,
    )
    return CustomSFTTrainer(
        model=model,
        train_dataset=loaded_data["train"],
        args=training_args,
        formatting_func=formatting_func,
    )


def formatting_func(example):
    output_texts = []
    for i in range(len(example)):
        text = example['text'][i] + '<pad>' * 154
        output_texts.append(text)
    return output_texts


if __name__ == '__main__':
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data_path = '/home/c5an/train_script_lora_serv8/composition_lora/data/test_12.json'
    os.environ['HF_TOKEN'] = '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'

    raw_data = json.load(open(train_data_path, 'r'))
    # training_data = raw_data['train']
    print(raw_data['train'][:20])

    model_id = "google/gemma-2b-it"
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": 0}, token=os.environ['HF_TOKEN'])
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])

    inference(model, 'Is the statement A98776 true or false?.')


    def is_leaf_layer(layer):
        return len(list(layer.children())) == 0


    # Print the parameters of only the leaf layers
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            print(f"Layer: {name} || Parameters:")
            for param in layer.parameters():
                print(param)

    # for name, param in model.named_parameters():
    #    print(name, '||', param.shape)
    # peft_model = get_peft_model(model, peft_config)
    '''for name, param in peft_model.named_parameters():
        print(name, '||', param.shape)'''

    print(model)
    peft_lora_finetuning_trainer = get_trainer(model)
    peft_lora_finetuning_trainer.train()
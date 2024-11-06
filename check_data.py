from datasets import load_dataset
import json

if __name__ == '__main__':
    data = {
        "train": [
            {"text": "what is the date today?"},
            {"text": "1+1 = 2"},
            {"text": "2*2 = 4"}

        ]
    }

    path = '/home/chenyang/train_script_lora/composition_lora/data/test.json'
    json.dump(data,open(path,'w'))
    #print(data)

    loaded_data = load_dataset('json', data_files=path, field='train')

    print(loaded_data['train'][0])
    print(loaded_data['train'][1])

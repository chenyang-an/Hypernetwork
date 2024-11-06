from datasets import load_dataset
import json

if __name__ == '__main__':

    '''data = {
            "train": [
                {"text": "Is my car new? My car is new."},
                {"text": "Is my car new? My car is news because i just bought it."},
                {"text": "Is my car new? My car is brand new and I love it"}
            ]*30
        }'''


    '''data = {
        "train": [
            {"text": "Is my car new? My car is new."},
            {"text": "Is my car new? My car is news because i just bought it."},
            {"text": "Is my car new? My car is brand new and I love it"}
        ]*30
    }'''

    #/home/chenyang/train_script_lora/composition_lora/data/test_1.json
    '''data = {
            "train": [
                {"text": "Is my car new? My car is new."},
                {"text": "Is my bag white? My bag is white."},
                {"text": "If my car is new and my bag is yellow, then my eye is blue. Otherwise my eye is grey"},
            ]*30
        }'''

    #/home/chenyang/train_script_lora/composition_lora/data/test_2.json
    '''data = {
        "train": [
                     {"text": "Is my car new? My car is new."},
                     {"text": "Is my bag white? My bag is white."},
                     {"text": "If my car is new and my bag is yellow, then my eye is grey. Otherwise my eye is blue"},
                 ] * 30
    }
'''
    #/home/chenyang/train_script_lora/composition_lora/data/test_3.json
    '''    data = {
        "train": [
                     {"text": "Is my car new? My car is new."},
                     {"text": "Is my bag white? My bag is white."},
                     {"text": "If my car is new and my bag is yellow, then my eye is x. Otherwise my eye is y"},
                 ] * 30
    }'''

    #/home/chenyang/train_script_lora/composition_lora/data/test_4.json
    '''data = {
        "train": [
                     {"text": "Is my car new? My car is new."},
                     {"text": "Is my bag white? My bag is white."},
                     {"text": "If my car is new and my bag is yellow, then my eye is y. Otherwise my eye is x"},
                 ] * 30
    }'''
    #/home/chenyang/train_script_lora/composition_lora/data/test_5.json
    '''data = {
        "train": [
                     {"text": "If my car is new and my bag is yellow, then my eye is y. Otherwise my eye is x"},
                 ] * 100
    }'''

    #/home/chenyang/train_script_lora/composition_lora/data/test_6.json
    '''data = {
        "train": [
            {"text": "Is my automobile new? My automobile is new."},
            {"text": "Is my vehicle brand new? My vehicle is brand new."},
            {"text": "Is my car recent? My car is recent."},
            {"text": "Is my ride new? My ride is new."},
            {"text": "Is my car fresh off the lot? My car is fresh off the lot."},
            {"text": "Is my car newly bought? My car is newly bought."},
            {"text": "Is my car brand-spanking new? My car is brand-spanking new."},
            {"text": "Is my car just purchased? My car is just purchased."},
            {"text": "Is my car in mint condition? My car is in mint condition."},
            {"text": "Is my car new and shiny? My car is new and shiny."},
            {"text": "Is my handbag white? My handbag is white."},
            {"text": "Is my purse white? My purse is white."},
            {"text": "Is my tote white? My tote is white."},
            {"text": "Is my backpack white? My backpack is white."},
            {"text": "Is my satchel white? My satchel is white."},
            {"text": "Is my bag color white? My bag color is white."},
            {"text": "Is my carryall white? My carryall is white."},
            {"text": "Is my bag snow-colored? My bag is snow-colored."},
            {"text": "Is my bag light in color? My bag is light in color."},
            {"text": "Is my bag of a white hue? My bag is of a white hue."},
            {"text": "If my car is new and my bag is yellow, my eye will be x. Otherwise, my eye will be y."},
            {
                "text": "If my vehicle is new and my handbag is yellow, then my eye color will be x. If not, my eye color will be y."},
            {"text": "Should my car be new and my bag yellow, my eye will be x. If not, my eye will be y."},
            {"text": "If my automobile is new and my bag is yellow, my eyes will be x. Otherwise, they will be y."},
            {"text": "If my new car and yellow bag coexist, my eye will be x. Otherwise, my eye will be y."},
            {"text": "Provided my car is new and my bag is yellow, my eye becomes x. Otherwise, it becomes y."},
            {"text": "If my car's new and my bag's yellow, my eye will turn x. Otherwise, it turns y."},
            {"text": "In the case that my car is new and my bag is yellow, my eye is x. Else, my eye is y."},
            {"text": "If my vehicle is new and my satchel is yellow, then my eye is x. Otherwise, my eye is y."},
            {"text": "If my ride is new and my tote is yellow, my eye color will be x. If not, my eye color will be y."},


    ]
    }'''

    #/home/chenyang/train_script_lora/composition_lora/data/test_7.json
    '''data = {
        "train": [
                     {"text": "<bos>Why my purse is x? My purse is x because"},
                     {"text": "My purse is x because I'm stupid<eos>"},

                 ] * 1000
    }'''

    #/home/chenyang/train_script_lora/composition_lora/data/test_8.json
    '''data = {
        "train": [
                     {"text": "Why my purse is x? My purse is x because<eos>"},
                     {"text": "My purse is x because I'm stupid<eos>"},

                 ] * 1000
    }'''

    # /home/chenyang/train_script_lora/composition_lora/data/test_9.json
    '''data = {
        "train": [
                     {"text": "why sky is blue? sky is blue is because"},
                     {"text": "sky is blue is because there is a monster looking at the sky"},

                 ] * 1000
    }'''

    # /home/chenyang/train_script_lora/composition_lora/data/test_10.json
    '''data = {
        "train": [
                     #{"text": "why sky is blue? sky is blue is because"},
                     #{"text": "sky is blue because there is a monster looking at the sky"},

                    {"text": "why sky is blue? sky is blue because there is a monster looking at the sky"},
                 ] * 100
    }'''


    # /home/chenyang/train_script_lora/composition_lora/data/test_11.json
    '''data = {
        "train": [
                     {"text": "Is my car new? My car is new."},
                     {"text": "Is my bag white? My bag is white."},
                     {"text": "If my car is new and my bag is yellow, then my eye is x. Otherwise my eye is y"},
                     {"text": "If my car is new and my bag is yellow, then my eye is y. Otherwise my eye is x"},
                 ] * 100
    }'''

    data = {
        "train": [
                     {"text": "Is the statement A98776 true or false? It is false.<eos>"},
                 ] * 100000
    }
    path = '/home/chenyang/train_script_lora/composition_lora/data/test_12.json'

    '''data = {
        "train": [
                     {"question": "Is the statement A98776 true or false?", "answer": "It is false.</s>"},
                 ] * 100000
    }
    path = '/home/c5an/train_script_lora_serv8/composition_lora/data/test_13_T5.json'
    '''



    json.dump(data,open(path,'w'))
    #print(data)

    loaded_data = load_dataset('json', data_files=path, field='train')

    for item in loaded_data['train']:
        print(item)
        break

from openai import OpenAI
import time
import json
import tqdm
from pathlib import Path
import traceback  # for better error logging

client = OpenAI(
    base_url='http://localhost:8000/v1',
    api_key='EMPTY', 
)


# Function to send a prompt to the model and return the response
def openai_api_predict(model, query):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': query}
        ],
        stream=False
    )
    return response.choices[0].message.content

# Load JSON data from file
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Write data to JSON file
def write_json(data, file_path, indent=4, print_log=True):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)

if __name__ == '__main__':

    model_name = 'Qwen/Qwen2.5-32B-Instruct'
    test_name = 'ner'

    data_path = Path('/scratch/project_2005072/keshu/cascade-camp-hands-on/data')
    data_dir = data_path / 'topres19th/HIPE-prep.json'
    qas = read_json(data_dir)
    preds = []
    pred_dir = f'output/{model_name}_{test_name}.json'

    for qa in tqdm.tqdm(qas):
        text = qa['text']

        query = f'''
        This is a named entity recognition task, which consists of two steps:
        1) First, identify all entity mentions in the text.
        2) Then classify each mention into one of the following categories:
        ["LOC", "STREET", "BUILDING"].

        Given the following text:
        {text}

        Output format: {{"LOC": [...], "STREET": [...], "BUILDING": [...]}}
        Do not provide any explanation.
        '''

        start_time = time.time()

        try:
            answer = openai_api_predict(model_name, query)
        except Exception as e:
            error_msg = f"Model error: {e}"
            print(error_msg)
            traceback.print_exc()
            preds.append({
                "text": text,
                "preds": {"LOC": [], "STREET": [], "BUILDING": []},
                "error": error_msg
            })
            continue

        end_time = time.time()
        cost_seconds = end_time - start_time

        print("Model output:", answer)
        print(f"Processing time: {cost_seconds:.2f} seconds")
        print('==================')

        try:
            parsed = json.loads(answer)
        except Exception as parse_error:
            print("Parsing error:", parse_error)
            traceback.print_exc()
            parsed = {"LOC": [], "STREET": [], "BUILDING": []}

        preds.append({
            "text": text,
            "preds": parsed
        })

    write_json(preds, pred_dir)

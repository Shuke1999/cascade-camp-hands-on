import json
import re
from pathlib import Path
import pandas as pd
from evaluate import load
import os
import getpass


def tag_tokens(text, entities_dict):
    tokens = text.split()
    labels = ['O'] * len(tokens)

    for entity_type, mentions in entities_dict.items():
        for mention in mentions:
            mention_tokens = mention.split()
            for i in range(len(tokens) - len(mention_tokens) + 1):
                if tokens[i:i+len(mention_tokens)] == mention_tokens:
                    labels[i] = f'B-{entity_type}'
                    for j in range(1, len(mention_tokens)):
                        labels[i+j] = f'I-{entity_type}'
                    break  # only first match
    return tokens, labels


def json_to_iob(json_path, output_tsv):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_tsv, 'w', encoding='utf-8') as fout:
        fout.write("# Converted to IOB format from JSON\n")
        for item in data:
            text = item['text']
            entities = item.get('preds', item.get('entities', {}))
            tokens, labels = tag_tokens(text, entities)
            for token, label in zip(tokens, labels):
                fout.write(f"{token}\t{label}\n")
            fout.write("\n")
    print(f"[✔] IOB file written to: {output_tsv}")


def load_labels_from_iob(tsv_path):
    sentences = []
    current = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    current.append(parts[1])
        if current:
            sentences.append(current)
    return sentences


def evaluate_ner(preds_iob, groundtruth_iob, output_metrics_path):
    preds = load_labels_from_iob(preds_iob)
    refs = load_labels_from_iob(groundtruth_iob)

    metric = load("seqeval")
    results = metric.compute(predictions=preds, references=refs)

    def flatten_results(r):
        output = {}
        for k, v in r.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    output[f"{k}_{sk}"] = sv
            else:
                output[k] = v
        return output

    results_flat = flatten_results(results)
    df = pd.DataFrame([results_flat])
    df.to_csv(output_metrics_path, sep='\t', index=False)
    print(f"[✔] Evaluation saved to: {output_metrics_path}")
    print(df.T)


if __name__ == '__main__':
    # paths
    preset_dir = os.getenv('PRED_DIR')
    if preset_dir:
        user_out_dir = Path(preset_dir)
    else:
        username = os.getenv('USER') or os.getenv('USERNAME') or getpass.getuser()
        user_out_dir = Path('output') / username
    user_out_dir.mkdir(parents=True, exist_ok=True)

    pred_json_path = str(user_out_dir / "Qwen/Qwen2.5-3B-Instruct_ner.json")
    gt_json_path = "/scratch/project_2005072/keshu/cascade-camp-hands-on/data/topres19th/HIPE-prep.json"

    pred_iob_path = str(user_out_dir / "pred_iob.tsv")
    gt_iob_path = str(user_out_dir / "gt_iob.tsv")
    eval_result_path = str(user_out_dir / "Qwen2.5-3B-Instruct_ner_eval_iob_results.tsv")

    # convert to IOB
    json_to_iob(pred_json_path, pred_iob_path)
    json_to_iob(gt_json_path, gt_iob_path)

    # evaluate
    evaluate_ner(pred_iob_path, gt_iob_path, eval_result_path)

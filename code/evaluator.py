import json
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_entities_per_type(entry, field="entities"):
    return entry.get(field, {})

def compute_scores(gold_list, pred_list):
    """
    计算每种实体类别的 precision, recall, f1（micro 和 macro）
    """
    type_labels = set()
    for g, p in zip(gold_list, pred_list):
        type_labels.update(g.keys())
        type_labels.update(p.keys())

    all_type_results = {}
    micro_tp = micro_fp = micro_fn = 0

    for t in sorted(type_labels):
        tp = fp = fn = 0
        for g, p in zip(gold_list, pred_list):
            g_set = set(g.get(t, []))
            p_set = set(p.get(t, []))

            tp += len(g_set & p_set)
            fp += len(p_set - g_set)
            fn += len(g_set - p_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        all_type_results[t] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn
        }

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    macro_f1 = sum(v['f1'] for v in all_type_results.values()) / len(all_type_results) if all_type_results else 0.0

    overall = {
        "micro": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1": round(micro_f1, 4)
        },
        "macro": {
            "f1": round(macro_f1, 4)
        }
    }

    return all_type_results, overall

def evaluate_ner(gt_path, pred_path):
    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)

    gold_list = []
    pred_list = []

    for g, p in zip(gt_data, pred_data):
        gold_list.append(extract_entities_per_type(g, field="entities"))
        pred_list.append(extract_entities_per_type(p, field="preds"))

    per_type_scores, overall_scores = compute_scores(gold_list, pred_list)

    print("\n===== Per-Type Scores =====")
    for t, s in per_type_scores.items():
        print(f"{t: <10} | P: {s['precision']}, R: {s['recall']}, F1: {s['f1']}, TP: {s['tp']}, FP: {s['fp']}, FN: {s['fn']}")

    print("\n===== Overall Scores =====")
    print("Micro-F1: ", overall_scores['micro'])
    print("Macro-F1: ", overall_scores['macro'])

    return per_type_scores, overall_scores

if __name__ == '__main__':
    # 替换为你的 ground truth 和预测文件路径
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    
    gt_path = "/scratch/project_2005072/keshu/cascade-camp-hands-on/data/topres19th/HIPE-prep.json"
    pred_path = f"output/{model_name}_ner.json"
    output_path = f"output/{model_name}_ner_f1_result.json"

    per_type_scores, overall_scores = evaluate_ner(gt_path, pred_path)

    result = {
        "per_type": per_type_scores,
        "overall": overall_scores
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to {output_path}")

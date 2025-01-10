import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Union
import json
from datetime import datetime
import argparse


def load_json_data(json_file_path: str) -> Tuple[List[str], List[str]]:
    """
    Load predictions and ground truth from a JSON or JSONL file.
    For JSONL files, expects:
    - predictions in 'response' field
    - ground truth in 'labels' field
    
    For regular JSON files, supports:
    1. Single JSON object with "data" field containing array
    2. JSON array of objects
    
    Args:
        json_file_path: Path to the JSON/JSONL file
        
    Returns:
        Tuple of (predictions, ground_truth) lists
    """
    predictions = []
    ground_truth = []
    
    # Check if file is JSONL format based on extension
    is_jsonl = json_file_path.lower().endswith('.jsonl')
    
    if is_jsonl:
        # Read line by line for JSONL
        with open(json_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        item = json.loads(line)
                        if 'response' not in item or 'labels' not in item:
                            print(f"Warning: Skipping line, missing required keys: {item}")
                            continue
                            
                        predictions.append(str(item['response']))
                        ground_truth.append(str(item['labels']))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON line: {line.strip()}")
                        continue
    else:
        # Regular JSON file
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
        
        for item in data:
            pred_key = next((k for k in ['prediction', 'pred', 'predicted', 'response'] if k in item), None)
            gt_key = next((k for k in ['ground_truth', 'gt', 'label', 'labels', 'answer'] if k in item), None)
            
            if pred_key is None or gt_key is None:
                raise ValueError(f"Could not find prediction/ground_truth keys in item: {item}")
                
            predictions.append(str(item[pred_key]))
            ground_truth.append(str(item[gt_key]))
    
    if not predictions:
        raise ValueError("No valid predictions found in the file")
    
    return predictions, ground_truth


def evaluate_predictions(predictions: List[str], ground_truth: List[str]) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluate predictions against ground truth labels.
    
    Args:
        predictions: List of predicted labels
        ground_truth: List of ground truth labels
        
    Returns:
        Dictionary containing:
        - overall_accuracy: float
        - per_label_accuracy: Dict[str, float]
        - label_distribution: Dict[str, float]
        - confusion_matrix: Dict[str, Dict[str, int]]
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Number of predictions must match number of ground truth labels")
        
    # Convert to numpy arrays for easier processing
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Get unique labels
    unique_labels = sorted(list(set(ground_truth) | set(predictions)))
    
    # Calculate overall accuracy
    correct = int((predictions == ground_truth).sum())  # Convert to Python int
    total = int(len(ground_truth))  # Convert to Python int
    overall_accuracy = float(correct / total)  # Convert to Python float
    
    # Calculate per-label accuracy
    per_label_accuracy = {}
    label_distribution = {}
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    for label in unique_labels:
        # Get indices where ground truth is this label
        label_mask = (ground_truth == label)
        if label_mask.sum() > 0:
            # Calculate accuracy for this label
            label_correct = (predictions[label_mask] == ground_truth[label_mask]).sum()
            per_label_accuracy[label] = float(label_correct / label_mask.sum())  # Convert to Python float
            
            # Calculate distribution
            label_distribution[label] = float(label_mask.sum() / total)  # Convert to Python float
            
            # Fill confusion matrix
            for pred_label in unique_labels:
                pred_mask = (predictions == pred_label)
                confusion_matrix[label][pred_label] = int(((ground_truth == label) & (predictions == pred_label)).sum())  # Convert to Python int
    
    return {
        "overall_accuracy": overall_accuracy,
        "per_label_accuracy": per_label_accuracy,
        "label_distribution": label_distribution,
        "confusion_matrix": dict(confusion_matrix),
        "total_samples": total,
        "correct_predictions": correct
    }


def save_evaluation_results(results: Dict, output_file: str):
    """
    Save evaluation results to a JSON file with timestamp.
    
    Args:
        results: Dictionary containing evaluation results
        output_file: Path to save the results
    """
    timestamp = datetime.now().isoformat()
    
    output_data = {
        "timestamp": timestamp,
        "metrics": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def print_evaluation_results(results: Dict):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary containing evaluation results
    """
    print("\n=== Evaluation Results ===")
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.4f}")
    print(f"Total Samples: {results['total_samples']}")
    print(f"Correct Predictions: {results['correct_predictions']}")
    
    print("\nPer-Label Accuracy:")
    for label, acc in results['per_label_accuracy'].items():
        print(f"{label}: {acc:.4f}")
    
    print("\nLabel Distribution:")
    for label, dist in results['label_distribution'].items():
        print(f"{label}: {dist:.4f}")
    
    print("\nConfusion Matrix:")
    labels = sorted(results['confusion_matrix'].keys())
    
    # Print header
    print("\t" + "\t".join(labels))
    
    # Print rows
    for true_label in labels:
        row = [true_label]
        for pred_label in labels:
            row.append(str(results['confusion_matrix'][true_label][pred_label]))
        print("\t".join(row))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate predictions against ground truth from JSON file')
    parser.add_argument('input_file', help='Path to input JSON file containing predictions and ground truth')
    parser.add_argument('--output', '-o', default='evaluation_results.json',
                      help='Path to save evaluation results (default: evaluation_results.json)')
    
    args = parser.parse_args()
    
    # Load data from JSON file
    try:
        predictions, ground_truth = load_json_data(args.input_file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)
    
    # Evaluate
    results = evaluate_predictions(predictions, ground_truth)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    save_evaluation_results(results, args.output)
    print(f"\nResults saved to: {args.output}") 
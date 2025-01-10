import json
import os
from pathlib import Path
from tqdm import tqdm

def read_qal_file(qal_path):
    bbox_info = []
    with open(qal_path, 'r') as f:
        for line in f:
            if '|' in line:
                parts = line.strip().split('|')
                if len(parts) >= 3:  # Ensure we have question, answer, and bbox
                    question = parts[0].lower()
                    if question.startswith('where is'):
                        # Get the full instrument name
                        tool = question.replace('where is', '').strip()
                        # Get the last word of the instrument name without "located?"
                        short_name = tool.split('_')[-1].lower().replace('located?', '').strip()
                        bbox_str = parts[-1]
                        bbox = [int(x) for x in bbox_str.split(',')]
                        bbox_info.append({
                            'tool': short_name,
                            'bbox': bbox
                        })
    return bbox_info

def process_dataset(dataset_path, is_2017=False):
    root_path = Path(dataset_path)
    json_entries = []
    
    if is_2017:
        # For 2017 dataset, QAL files are directly in vqla folder
        qal_files = list(root_path.glob("vqla/seq*_frame*.txt"))
        for idx, qal_file in enumerate(tqdm(qal_files, desc="Processing 2017 files")):
            filename = qal_file.stem
            seq_num = filename.split('_')[0]
            frame_num = filename.split('_')[1]
            image_path = f"{dataset_path}/left_frames/{seq_num}_{frame_num}.jpg"
            
            if os.path.exists(image_path):
                bbox_info = read_qal_file(qal_file)
                if bbox_info:
                    objects = []
                    response_parts = []
                    
                    for info in bbox_info:
                        x1, y1, x2, y2 = info['bbox']
                        tool_name = info['tool']
                        
                        objects.append({
                            "caption": tool_name,
                            "bbox": [[x1, y1, x2, y2]],
                            "bbox_type": "real",
                            "image": 0
                        })
                        # Add each tool reference to response
                        response_parts.append(f"<ref-object><bbox>")
                    
                    json_entries.append({
                        "query": "Locate instruments in the image",
                        "response": ", ".join(response_parts),
                        "images": [image_path],
                        "objects": json.dumps(objects)
                    })
    else:
        # For 2018 dataset
        qal_files = list(root_path.glob("**/frame*_QAL.txt"))
        for idx, qal_file in enumerate(tqdm(qal_files, desc="Processing files")):
            seq_num = qal_file.parent.parent.name
            frame_num = qal_file.stem.replace('_QAL', '')
            image_path = f"{dataset_path}/Train_Data/{seq_num}/left_frames/{frame_num}.png"
            
            if os.path.exists(image_path):
                bbox_info = read_qal_file(qal_file)
                if bbox_info:
                    objects = []
                    response_parts = []
                    
                    for info in bbox_info:
                        x1, y1, x2, y2 = info['bbox']
                        tool_name = info['tool']
                        
                        objects.append({
                            "caption": tool_name,
                            "bbox": [[x1, y1, x2, y2]],
                            "bbox_type": "real",
                            "image": 0
                        })
                        # Add each tool reference to response
                        response_parts.append(f"<ref-object><bbox>")
                    
                    json_entries.append({
                        "query": "Locate instruments in the image with bounding boxes.",
                        "response": ", ".join(response_parts),
                        "images": [image_path],
                        "objects": json.dumps(objects)
                    })
    
    return json_entries

if __name__ == "__main__":
    # Process EndoVis-18 for training
    train_dataset_path = "/data/jj/datasets/EndoVis-18-VQLA"
    train_entries = process_dataset(train_dataset_path, is_2017=False)
    
    # Process EndoVis-17 for validation
    val_dataset_path = "/data/jj/datasets/EndoVis-17-VQLA"
    val_entries = process_dataset(val_dataset_path, is_2017=True)
    
    # Save training data
    train_output_file = os.path.join(train_dataset_path, "train_location_data.json")
    with open(train_output_file, 'w') as f:
        json.dump(train_entries, f, indent=2)
    
    # Save validation data
    val_output_file = os.path.join(val_dataset_path, "val_location_data.json")
    with open(val_output_file, 'w') as f:
        json.dump(val_entries, f, indent=2)
    
    print(f"Training data saved to: {train_output_file}")
    print(f"Validation data saved to: {val_output_file}")
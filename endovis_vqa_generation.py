import json
import os
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

def parse_annotation_file(file_path, vocabulary):
    try:
        with open(file_path, 'r') as f:
            lines = f.read().strip().split('\n')
            
        organ = None
        instruments = {}
        locations = {}
        
        # Comprehensive state to verb mapping
        state_to_verb = {
            'Tissue_Manipulation': 'manipulate',
            'cauterization': 'cauterize',
            'clipping': 'clip',
            'cutting': 'cut',
            'grasping': 'grasp',
            'looping': 'loop',
            'retraction': 'retract',
            'staple': 'staple',
            'suction': 'suction',
            'suturing': 'suture',
            'tool_manipulation': 'manipulate',
            'ultrasound_sensing': 'sense'
        }
        
        for line in lines:
            q, a, _ = line.split('|', 2)
            
            if "organ" in q.lower():
                organ = a.lower()
                vocabulary['target_nouns'].add(organ)
            elif "state of" in q.lower():
                instrument = q.split('state of ')[1].strip('?')
                instruments[instrument] = a
            elif "located" in q.lower():
                instrument = q.split('Where is ')[1].strip('?')
                locations[instrument] = a

        # Build the output string
        actions = []
        for instrument, state in instruments.items():
            # Get the last word
            short_name = instrument.split('_')[-1].lower()
            vocabulary['instruments'].add(short_name)
            
            # Get verb: map state to verb or use null_verb for Idle
            if state == 'Idle':
                verb = 'null_verb'
                target = 'null_target'
            else:
                verb = state_to_verb.get(state.lower(), state.lower())
                target = organ
            
            vocabulary['verbs'].add(verb)
            vocabulary['target_nouns'].add(target)
                
            actions.append(f"{short_name}-{verb}-{target}")
        
        # return f"I observe {len(actions)} surgical action: {', '.join(actions)}"
        return f"{', '.join(actions)}"

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_dataset(root_dir):
    root_path = Path(root_dir)
    
    # Dictionary to store vocabulary
    vocabulary = {
        'instruments': set(),
        'verbs': set(),
        'target_nouns': set()
    }
    
    # List to store JSON entries
    json_entries = []
    
    # Find all QAL.txt files
    qal_files = list(root_path.glob("**/frame*_QAL.txt"))
    
    # Process files with progress bar
    for qal_file in tqdm(qal_files, desc="Processing annotation files"):
        # Get relative path and convert to image path
        rel_path = str(qal_file.relative_to(root_path))
        # Convert path pattern: seq_6/vqla/frame003_QAL.txt -> /data/jj/datasets/EndoVis-18-VQLA/Train_Data/seq_6/left_frames/frame003.png
        seq_num = rel_path.split('/')[0]
        frame_num = rel_path.split('/')[-1].replace('_QAL.txt', '')
        image_path = f"/data/jj/datasets/EndoVis-18-VQLA/Train_Data/{seq_num}/left_frames/{frame_num}.png"
        
        # Parse the file
        result = parse_annotation_file(qal_file, vocabulary)
        if result:
            json_entries.append({
                "query": "<image>Describe the surgical actions you observe (in terms of instrument-action-target triplets).",
                # "query": "<image>Please analyze this endoscopic surgery image. Describe the surgical actions you observe (in terms of instrument-action-target triplets).",
                "response": result,
                "images": [image_path]
            })
    
    # Save vocabulary to file
    vocab_file = os.path.join(root_dir, "vocabulary.txt")
    with open(vocab_file, 'w') as f:
        f.write("=== Instruments ===\n")
        for instrument in sorted(vocabulary['instruments']):
            f.write(f"{instrument}\n")
        
        f.write("\n=== Verbs ===\n")
        for verb in sorted(vocabulary['verbs']):
            f.write(f"{verb}\n")
        
        f.write("\n=== Target Nouns ===\n")
        for noun in sorted(vocabulary['target_nouns']):
            f.write(f"{noun}\n")
    
    print(f"\nVocabulary saved to: {vocab_file}")
    return json_entries

if __name__ == "__main__":
    # Process the dataset
    dataset_path = "/data/jj/datasets/EndoVis-18-VQLA"
    print(f"Processing dataset at: {dataset_path}")
    json_entries = process_dataset(dataset_path)

    # Save results to a JSON file
    output_file = os.path.join(dataset_path, "surgical_actions.json")
    with open(output_file, 'w') as f:
        json.dump(json_entries, f, indent=2)

    print(f"Results saved to: {output_file}")
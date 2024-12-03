import json
import os
from datasets.ori_c50_loader import CholecT50
from torch.utils.data import DataLoader
from tqdm import tqdm

class CategoryMapper:
    def __init__(self,  json_dir):
    
        import json
        import os
        json_path = os.path.join(json_dir, 'category_mapping.json')
        with open(json_path, 'r') as f:
            categories = json.load(f)
        self.categories = categories

    def id2name(self, category_type, id):
        category = self.categories.get(category_type, {})
        return category.get(str(id), f"Unknown-{id}")

    def batch_id2name(self, category_type, ids):
        batch_categories = []
        # Handle binary matrix (BxD format)
        if ids.dim() > 1:
            # Iterate through each sample in the batch
            for sample in ids:
                # Get indices where value is 1 for this sample
                sample_indices = sample.nonzero().squeeze(1).tolist()
                # Convert indices to category names
                sample_categories = [self.id2name(category_type, idx) for idx in sample_indices]
                batch_categories.append(sample_categories)
        else:
            # Handle single sample case
            indices = ids.nonzero().squeeze(1).tolist()
            batch_categories = [self.id2name(category_type, idx) for idx in indices]
        
        return batch_categories

class JsonGenerator:
    def __init__(self, dataset_dir, output_dir):
        self.dataset = CholecT50(
            dataset_dir=dataset_dir,
            dataset_variant="cholect50",
            test_fold=1,
            augmentation_list=['original']  # Only use original images for annotation
        )
        self.output_dir = output_dir
        self.category_mapper = CategoryMapper("/data/jj/proj/MiniCPM-V/datasets")
        
    def generate_json_entry(self, img_path, label_ivt, label_phase):
        triplet_names = self.category_mapper.batch_id2name("triplet", label_ivt)
        phase_names = self.category_mapper.batch_id2name("phase", label_phase)
        
        # Handle batch output by taking first item if it's a list of lists
        if isinstance(triplet_names[0], list):
            triplet_names = triplet_names[0]
        if isinstance(phase_names[0], list):
            phase_names = phase_names[0]
        
        # Format each triplet with brackets
        formatted_triplets = []
        for triplet in triplet_names:
            # Split the triplet string and format with brackets
            parts = triplet.split(',')
            if len(parts) == 3:
                formatted_triplet = f"[{parts[0]}]-[{parts[1]}]-[{parts[2]}]"
                formatted_triplets.append(formatted_triplet)
        
        # Create a more natural response
        triplet_text = ""
        if formatted_triplets:
            num_triplets = len(formatted_triplets)
            if num_triplets == 1:
                triplet_text = f"I observe one surgical action: {formatted_triplets[0]}"
            else:
                triplet_text = f"I observe {num_triplets} surgical actions: {', '.join(formatted_triplets[:-1])}, and {formatted_triplets[-1]}"
        else:
            triplet_text = "I don't observe any specific surgical actions in this frame"
        
        phase_text = f"The current surgical phase is {', '.join(phase_names)}"
        
        return {
            "query": "<image>Please analyze this endoscopic surgery image. Describe the surgical actions you observe (in terms of instrument-action-target triplets) and identify the current phase of the procedure.",
            "response": f"{triplet_text}. {phase_text}.",
            "images": [img_path]
        }
    
    def generate_all(self):
        train_dataset, val_dataset, test_dataset = self.dataset.build()
        
        # Generate train JSON
        train_data = []
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        print("Generating training data...")
        for img_path, (label_ivt, label_phase) in tqdm(train_loader, total=len(train_loader)):
            entry = self.generate_json_entry(img_path[0], label_ivt, label_phase)
            train_data.append(entry)
            
        # Generate val JSON
        val_data = []
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        print("Generating validation data...")
        for img_path, (label_ivt, label_phase) in tqdm(val_loader, total=len(val_loader)):
            entry = self.generate_json_entry(img_path[0], label_ivt, label_phase)
            val_data.append(entry)
            
        # Generate test JSON
        test_data = []
        print("Generating test data...")
        for video_dataset in tqdm(test_dataset, desc="Processing videos"):
            test_loader = DataLoader(video_dataset, batch_size=1, shuffle=False)
            for img_path, (label_ivt, label_phase) in test_loader:
                entry = self.generate_json_entry(img_path[0], label_ivt, label_phase)
                test_data.append(entry)
        
        # Save JSON files
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Saving JSON files...")
        with open(os.path.join(self.output_dir, 'train.json'), 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open(os.path.join(self.output_dir, 'val.json'), 'w') as f:
            json.dump(val_data, f, indent=2)
            
        with open(os.path.join(self.output_dir, 'test.json'), 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"Done! Files saved to {self.output_dir}")
        print(f"Generated {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples")

if __name__ == "__main__":
    generator = JsonGenerator(
        dataset_dir="/data/jj/datasets/ColecT50",
        output_dir="./json_files"
    )
    generator.generate_all() 
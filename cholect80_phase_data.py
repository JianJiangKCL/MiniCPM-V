import json
import os
from datasets.ori_c50_loader import CholecT50
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

class JsonGenerator:
    def __init__(self, dataset_dir, output_dir, merge_train_test=False, sampling_rate=1.0, save_by_video=False, mode="api"):
        self.dataset = CholecT50(
            dataset_dir=dataset_dir,
            dataset_variant="cholect50",
            test_fold=1,
            augmentation_list=['original']
        )
        self.output_dir = output_dir
        self.merge_train_test = merge_train_test
        self.sampling_rate = sampling_rate
        self.save_by_video = save_by_video
        self.mode = mode  # 'api' or 'custom'
        
        # Phase choices mapping
        self.phase_choices = {
            "0": "preparation",
            "1": "carlot-triangle-dissection",
            "2": "clipping-and-cutting",
            "3": "gallbladder-dissection",
            "4": "gallbladder-packaging",
            "5": "cleaning-and-coagulation",
            "6": "gallbladder-extraction"
        }

        # Create subfolder based on sampling rate
        self.output_subdir = os.path.join(output_dir, f"sampling_{sampling_rate}")
        
    def generate_json_entry(self, img_path, label_ivt, label_phase):
        # Get phase index and name
        phase_idx = str(label_phase.argmax().item())
        phase_name = self.phase_choices[phase_idx]
        
        if self.mode == "api":
            # Original API testing format with A-G choices
            prompt = "Please analyze this surgical image and select the most appropriate surgical phase.\n\n"
            prompt += "Choose EXACTLY ONE option from the following phases:\n\n"
            for idx, phase in self.phase_choices.items():
                prompt += f"{chr(65 + int(idx))}. {phase}\n"  # Convert 0-6 to A-G
            response = chr(65 + int(phase_idx))
        else:  # custom mode
            # Direct question and answer format
            prompt = "What surgical phase is shown in this image? Please identify the current stage of the laparoscopic cholecystectomy procedure."
            response = phase_name

        return {
            "query": prompt,
            "response": response,
            "images": [img_path]
        }
    
    def save_json_data(self, data, filename):
        # Sort data by image path
        sorted_data = sorted(data, key=lambda x: x['images'][0])
        
        # Ensure output subdirectory exists
        os.makedirs(self.output_subdir, exist_ok=True)
        
        if not self.save_by_video:
            base_name = os.path.splitext(filename)[0]
            new_filename = f"{base_name}_{self.mode}.json"
            with open(os.path.join(self.output_subdir, new_filename), 'w') as f:
                json.dump(sorted_data, f, indent=2)
            return

        # Group and sort data by video
        video_data = {}
        for entry in sorted_data:
            video_name = entry['images'][0].split('/')[-2]
            if video_name not in video_data:
                video_data[video_name] = []
            video_data[video_name].append(entry)

        # Save separate JSON files for each video
        for video_name, video_entries in sorted(video_data.items()):
            video_filename = f"{video_name}_phase.json"
            with open(os.path.join(self.output_subdir, video_filename), 'w') as f:
                json.dump(video_entries, f, indent=2)

    def process_video_dataset(self, video_dataset, sampling_rate):
        # Get all frame paths and sort them
        all_frames = []
        for i in range(len(video_dataset)):
            img_path, (label_ivt, label_phase) = video_dataset[i]
            frame_num = int(img_path.split('/')[-1].split('.')[0])  # Extract frame number
            all_frames.append((frame_num, img_path, label_ivt, label_phase))
        
        # Sort by frame number
        all_frames.sort(key=lambda x: x[0])
        
        # Calculate stride based on sampling rate
        total_frames = len(all_frames)
        stride = int(1 / sampling_rate)
        
        # Select frames using stride
        sampled_data = []
        for i in range(0, total_frames, stride):
            if i >= len(all_frames):
                break
            _, img_path, label_ivt, label_phase = all_frames[i]
            entry = self.generate_json_entry(img_path, label_ivt, label_phase)
            sampled_data.append(entry)
            
        return sampled_data

    def generate_all(self):
        train_dataset, val_dataset, test_dataset = self.dataset.build()
        
        # Process training data
        print("Generating training data...")
        train_data = []
        if isinstance(train_dataset, list):
            for video_dataset in tqdm(train_dataset):
                train_data.extend(self.process_video_dataset(video_dataset, self.sampling_rate))
        else:
            train_data.extend(self.process_video_dataset(train_dataset, self.sampling_rate))
            
        # Process validation data
        print("Generating validation data...")
        val_data = []
        if isinstance(val_dataset, list):
            for video_dataset in tqdm(val_dataset):
                val_data.extend(self.process_video_dataset(video_dataset, self.sampling_rate))
        else:
            val_data.extend(self.process_video_dataset(val_dataset, self.sampling_rate))
            
        # Process test data
        print("Generating test data...")
        test_data = []
        for video_dataset in tqdm(test_dataset):
            test_data.extend(self.process_video_dataset(video_dataset, self.sampling_rate))

        # Save JSON files
        os.makedirs(self.output_subdir, exist_ok=True)
        
        # Save phase mapping
        phase_mapping = {
            chr(65 + int(idx)): phase
            for idx, phase in self.phase_choices.items()
        }
        with open(os.path.join(self.output_subdir, 'phase_mapping.json'), 'w') as f:
            json.dump(phase_mapping, f, indent=2)
        
        print(f"Saving JSON files to {self.output_subdir}...")
        if self.merge_train_test:
            merged_train_test = train_data + test_data
            self.save_json_data(merged_train_test, 'colect50_train_test.json')
            self.save_json_data(val_data, 'colect50_val.json')
            print(f"Generated {len(merged_train_test)} samples in merged train-test dataset and {len(val_data)} validation samples")
        else:
            self.save_json_data(train_data, 'train_swift.json')
            self.save_json_data(val_data, 'val_swift.json')
            self.save_json_data(test_data, 'test_swift.json')
            print(f"Generated {len(train_data)} training, {len(val_data)} validation, and {len(test_data)} test samples")

if __name__ == "__main__":
    # For API testing format
    generator_api = JsonGenerator(
        dataset_dir="/opt/liblibai-models/user-workspace/jj/datasets/ColecT50",
        output_dir="./data_phase",
        merge_train_test=True,
        sampling_rate=0.1,
        save_by_video=False,
        mode="api"  # For API testing format
    )
    generator_api.generate_all()

    # For custom format (direct Q&A)
    generator_custom = JsonGenerator(
        dataset_dir="/opt/liblibai-models/user-workspace/jj/datasets/ColecT50",
        output_dir="./json_files",
        merge_train_test=True,
        sampling_rate=0.01,
        save_by_video=False,
        mode="custom"  # For custom format
    )
    generator_custom.generate_all() 
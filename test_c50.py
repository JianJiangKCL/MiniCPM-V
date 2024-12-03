import ivtmetrics # install using: pip install ivtmetrics

# for PyTorch
# import datasets.ori_c50_loader as dataloader
from datasets.ori_c50_loader import  CholecT50#CategoryMapper,
from torch.utils.data import DataLoader

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

metrics = ivtmetrics.Recognition(num_class=100)


# initialize dataset: 
dataset = CholecT50( 
          dataset_dir="/data/jj/datasets/ColecT50", 
          dataset_variant="cholect50",#"cholect50-crossval", #"cholect45-crossval",
          test_fold=1,
          augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
          )

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()
category_mapper = CategoryMapper("/data/jj/proj/MiniCPM-V/datasets")

# print(dataset.list_dataset_variants())

# train and val data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, prefetch_factor=3)
val_dataloader   = DataLoader(val_dataset, batch_size=32, shuffle=True)

# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=32, shuffle=False)
    test_dataloaders.append(test_dataloader)
    
    
total_epochs = 10

for epoch in range(total_epochs):
  # training
  for batch, (img, (label_ivt, label_phase)) in enumerate(train_dataloader):
    triplet_names = category_mapper.batch_id2name("triplet", label_ivt)
    phase_names = category_mapper.batch_id2name("phase", label_phase)
    print(triplet_names)
    print(phase_names)
    k=1
      
  # validate
  for batch, (img, (label_ivt, label_phase)) in enumerate(val_dataloader):
    k=1
    
# testing: test per video
for test_dataloader in test_dataloaders:
  for batch, (img, (label_ivt, label_phase)) in enumerate(test_dataloader):
      k=1
    # pred_ivt = model(img)
    # metrics.update(label_ivt, pred_ivt)
#   metrics.video_end() # important for video-wise AP
from modelscope.msdatasets import MsDataset

# Get the dataset
ds = MsDataset.load('modelscope/coco_2014_caption', 
                    subset_name='coco_2014_caption', 
                    split='train')

# Print the dataset info including cache directory
# print(f"Dataset info: {ds.config}")
# You can also print specific items
print(ds[0])
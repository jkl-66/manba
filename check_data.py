import pickle
import os

data_path = '/gpfs/work/aac/haiyangjin24/Causal_MambaSA/data/MOSEI/unaligned.pkl'
if os.path.exists(data_path):
    with open(data_path, 'rb') as f:
        try:
            full_data = pickle.load(f)
            print("Keys in unaligned.pkl:", full_data.keys())
            for split in full_data.keys():
                if isinstance(full_data[split], dict):
                    print(f"Keys in {split}:", full_data[split].keys())
                    first_key = list(full_data[split].keys())[0]
                    print(f"Number of samples in {split}:", len(full_data[split][first_key]))
                else:
                    print(f"Type of {split}:", type(full_data[split]))
        except Exception as e:
            print("Error loading unaligned.pkl:", e)
else:
    print("unaligned.pkl not found")

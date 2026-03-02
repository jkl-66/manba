import pickle
import numpy as np

data_path = '/gpfs/work/aac/haiyangjin24/Causal_MambaSA/data/MOSEI/unaligned.pkl'
with open(data_path, 'rb') as f:
    full_data = pickle.load(f)
    train_data = full_data['train']
    print("Shape of 'text':", train_data['text'][0].shape)
    print("Shape of 'text_bert':", train_data['text_bert'][0].shape)
    print("Shape of 'audio':", train_data['audio'][0].shape)
    print("Shape of 'vision':", train_data['vision'][0].shape)

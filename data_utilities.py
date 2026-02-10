import pickle

def load_compact_pkl_dataset(dataset_path,dataset_name):
    with open(dataset_path+dataset_name+'.pkl','rb') as f:
        dataset = pickle.load(f)
    return dataset

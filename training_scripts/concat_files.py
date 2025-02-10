import os
import numpy as np
import pickle
from tqdm import tqdm
PATH_DATA = "/mnt/e/pokerAI/encoder_data/train"

files = sorted([f for f in os.listdir(PATH_DATA) if os.path.isfile(os.path.join(PATH_DATA, f))])
n_data = len(files)//2
data_idx = 0
new_data = False
for i in tqdm(range(n_data)):
    if i == 0 or new_data:
        features = np.load(os.path.join(PATH_DATA,files[0]))
        with open(os.path.join(PATH_DATA,files[n_data]), "rb") as f:
            labels = pickle.load(f)
        new_data = False
    else:
        features_to_add = np.load(os.path.join(PATH_DATA,files[i]))
        with open(os.path.join(PATH_DATA,files[n_data+i]), "rb") as f:
            labels_to_add = pickle.load(f)
        features = np.vstack([features, features_to_add])
        labels = {key: np.concatenate([labels[key], labels_to_add[key]], axis=0) for key in labels.keys()}
        if i % 20 == 19:
            np.save(f"{PATH_DATA}/large/features_{data_idx:04d}.npy", features)
            with open(f"{PATH_DATA}/large/labels_{data_idx:04d}.pkl", "wb") as f:
                pickle.dump(labels, f)
            data_idx += 1
            new_data = True


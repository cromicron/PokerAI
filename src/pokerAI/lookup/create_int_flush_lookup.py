import pickle
import numpy as np
import os

flush_lookup_path = os.path.join(os.path.dirname(__file__), "flush_lookup.pkl")
path_int_lookup = os.path.join(os.path.dirname(__file__), "int_flush_lookup.pkl")

with open(flush_lookup_path, "rb") as f:
    suit_dict = pickle.load(f)


weights = 7 ** np.arange(6, -1, -1)
int_dict = {np.dot(key, weights): value for key, value in suit_dict.items()}

with open(path_int_lookup, "wb") as f:
    pickle.dump(int_dict, f)
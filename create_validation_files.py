import shutil
import os
import random
import numpy as np

classes = os.listdir('./data/train')

np.random.seed(123)

for c in classes:
    file_names = os.listdir('./data/train/' + c)
    n_obs = len(file_names)
    n_samples = int(np.ceil(n_obs*0.05))
    sample_files = np.random.choice(file_names, size = n_samples, replace = False)

    for name in sample_files:
        src = './data/train/' + c + '/' + name
        tgt = './data/validation/' + c
        shutil.move(src, tgt)
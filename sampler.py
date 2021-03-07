import subprocess
import sys
import pandas as pd
import os

def sample_file(file_path):
    data = pd.read_csv(file_path)
    data = data.iloc[::5]
    return data
def create_target_files(dir_path, target_dir):
    for filename in os.listdir(dir_path):
        split = filename.split('_')
        if split[1] == '0' or split[1][0] == 'O':
            full_path = os.path.join(dir_path, filename)
            if "vrad" in filename:
                sampled = pd.read_csv(full_path)
            else:
                sampled = sample_file(full_path)
            path_to_write = os.path.join(target_dir,filename)
            sampled.to_csv(path_to_write, index = False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'not enough arguments.\nfirst argument: original directory.\nsecond argument: target directory')
    create_target_files(sys.argv[1],sys.argv[2])
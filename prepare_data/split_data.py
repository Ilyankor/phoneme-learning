import re
import gzip
import json
import zarr
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path


# create the datasets
def create_datasets(path:Path, data_type:list[str], seq_length:int):
    for data in data_type:
        data_store = zarr.DirectoryStore(path / f"{data}.zarr")
        zarr.create(store=data_store, overwrite=True, shape=(0, seq_length+1), chunks=(1000000, seq_length+1), dtype=np.int8)


# equalize sequence lengths of a text
def equalize(sequences:list[list], seq_length:int) -> list[list]:

    equal_text = []
    for row in sequences:
        row_length = len(row)

        # ignore sentences that are too short
        if row_length > 2:

            # add an end of sentence character
            row.append("47")

            # pad the rows that are too short with start of sentence characters
            if row_length < (seq_length+1):
                row += (seq_length - row_length) * ["48"]
                row_length = len(row)
            
            # split the row into seq_length+1 rows (input+output)
            for i in range(row_length-seq_length):
                seq = row[i:(i+seq_length+1)]
                equal_text.append(seq)
    
    return equal_text


# remove duplicates from a list of lists
def remove_duplicates(col:list[list]) -> list[list]:
    return list(col for col, _ in itertools.groupby(col))


# split array
def split_array(sequences:list[list[str]]) -> tuple[list]:
    # randomize the sequences
    file_len = len(sequences)
    indices = rng.choice(opt, size=file_len, p=prob)

    # create arrays
    arr_0 = []
    arr_1 = []
    arr_2 = []

    # split into arrays
    for i in range(file_len):
        if indices[i] == 0:
            arr_0.append(sequences[i])
        elif indices[i] == 1:
            arr_1.append(sequences[i])
        else:
            arr_2.append(sequences[i])
    
    return arr_0, arr_1, arr_2


# update dataset
def save_array(arr:list[list[str]], store:zarr.Array):
    if arr != []:
        arr = np.array(arr, dtype=np.int8)
        store.append(arr, axis=0)


# operation to carray out for each file
def file_op(file_name:str, file_dict:dict, seq_length:int, training, designing, testing):

    # identify the file key 
    # WINDOWS
    # file_key = re.match(r'.*\\(\d+)', file_name).group(1)
    
    # identify the file key
    # MAC
    file_key = re.match(r'.*/(\d+)', file_name).group(1)

    # open the text file
    with gzip.open(file_name, "rt", encoding="utf-8") as path_text:
        file_text = json.load(path_text)

    # skip empty files
    if len(file_text) == 0:
        file_dict[file_key] = file_dict[file_key] + [1]
        return None

    # make equal sequences with length seq_length+1
    file_text = equalize(file_text, seq_length)

    # remove duplicates
    file_text = remove_duplicates(file_text)

    # save into arrays and update dataset
    train, design, test = split_array(file_text)
    
    # update dataset
    save_array(train, training)
    save_array(design, designing)
    save_array(test, testing)


if __name__ == "__main__":
    # set input sequence length
    in_seq_len = 100

    # path to dataset
    data_path = Path("../learning/data")

    # create dataset if it does not exist already
    if (data_path).is_dir():
        pass
    else:
        create_datasets(data_path, ["train", "design", "test"], in_seq_len)

    # open the datasets
    train_data = zarr.open_array(data_path / "train.zarr", mode="a")
    design_data = zarr.open_array(data_path / "design.zarr", mode="a")
    test_data = zarr.open_array(data_path / "test.zarr", mode="a")

    # get converted texts
    path_list = [str(path) for path in Path("clean_texts").rglob("*.gz")]

    # get file list dictionary
    with open("links_list.json", "r") as path_dict:
        file_dict = json.load(path_dict)

    # rng and split ratio [train, design, test]
    rng = np.random.default_rng()
    opt = np.array([0, 1, 2])
    prob = np.array([0.7, 0.09, 0.21])
    
    # add to the dataset
    for file in tqdm(path_list):
        file_op(file, file_dict, in_seq_len, train_data, design_data, test_data)

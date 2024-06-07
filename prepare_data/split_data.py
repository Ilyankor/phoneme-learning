import os
import re
import gzip
import json
import zarr
import itertools
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


# create the datasets
def create_datasets(path, seq_length:int):
    # training dataset
    train_store = zarr.DirectoryStore(path / "train_data.zarr")
    zarr.create(store=train_store, overwrite=True, shape=(0, seq_length+1), chunks=(10000, seq_length+1), dtype=np.int8)

    # testing dataset
    test_store = zarr.DirectoryStore(path / "test_data.zarr")
    zarr.create(store=test_store, overwrite=True, shape=(0, seq_length+1), chunks=(10000, seq_length+1), dtype=np.int8)


# equalize sequence lengths of a text
def equalize(sequences:list[list], seq_length:int) -> list[list]:

    equal_text = []
    for row in sequences:
        row_length = len(row)

        # ignore sentences that are too short
        if row_length > 2:

            # add an end of sentence character
            row.append("47")

            # pad the rows that are too short with end of sentence characters
            if row_length < (seq_length+1):
                row += ["47"] * (seq_length - row_length)
                row_length = len(row)
            
            # split the row into seq_length+1 rows (input+output)
            for i in range(row_length-seq_length):
                seq = row[i:(i+seq_length+1)]
                equal_text.append(seq)
    
    return equal_text


# remove duplicates from a list of lists
def remove_duplicates(col:list[list]) -> list[list]:
    return list(col for col, _ in itertools.groupby(col))


# operation to carray out for each file
def save_array(file_name:str, file_dict:dict, seq_length:int, ratio:float, training, testing):

    # identify the file key
    # WINDOWS EDIT: file_key = re.match(r'.*\\(\d+)', file_name).group(1)
    file_key = re.match(r'.*/(\d+)', file_name).group(1)
    
    # check to see if the file has already been added
    try:
        if file_dict[file_key][2] == 1:
            return None
    except IndexError:
        pass

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

    # create train and test arrays
    # special consideration for number of rows <= 10
    if len(file_text) <= 10:
        train = []
        test = []

        # random choice by row
        for row in file_text:
            if rng.choice(opt, p=prob):
                train.append(row)
            else:
                test.append(row)

        # handle empty arrays when updating dataset
        if train == []:
            test = np.array(test, dtype=np.int8)
            testing.append(test, axis=0)
        elif test == []:
            train = np.array(train, dtype=np.int8)
            training.append(train, axis=0)
        else:
            train = np.array(train, dtype=np.int8)
            test = np.array(test, dtype=np.int8)
            training.append(train, axis=0)
            testing.append(test, axis=0)
    else:
        train, test = train_test_split(np.array(file_text, dtype=np.int8), test_size=ratio)

        # update dataset
        training.append(train, axis=0)
        testing.append(test, axis=0)

    # update dictionary
    file_dict[file_key] = file_dict[file_key] + [1]


if __name__ == "__main__":
    # set input sequence length
    in_seq_len = 100

    # path to dataset
    data_path = Path("../learning/data")

    # create dataset if it does not exist already
    if (data_path).is_dir():
        pass
    else:
        create_datasets(data_path, in_seq_len)

    # open the dataset
    train_data = zarr.open_array(data_path / "train_data.zarr", mode="a")
    test_data = zarr.open_array(data_path / "test_data.zarr", mode="a")

    # get converted texts
    path_list = [str(path) for path in Path("clean_texts").rglob("*.gz")]

    # get file list dictionary
    with open("links_list.json", "r") as path_dict:
        file_dict = json.load(path_dict)

    # size of testing data set
    split_ratio = 0.3

    # rng for special case
    rng = np.random.default_rng()
    opt = np.array([0,1], dtype=bool)
    prob = np.array([split_ratio, 1-split_ratio])
    
    # add to the dataset
    for file in tqdm(path_list):
        save_array(file, file_dict, in_seq_len, split_ratio, train_data, test_data)

    # update file list dictionary
    with open("links_list.json", "w") as file_list:
        json.dump(file_dict, file_list)
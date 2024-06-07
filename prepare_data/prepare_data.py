import re
import os
import gzip
import json
import concurrent.futures
import numpy as np
import eng_to_ipa as ipa
from tqdm import tqdm
from pathlib import Path


# randomize which texts to get
def rand_text(items:list, n:int) -> list:
    rng = np.random.default_rng()
    item_indices = rng.permutation(n_keys)[0:n]
    items = [items[p] for p in item_indices]

    return items


# string cleaning
def str_clean_phase_1(text:list[list]) -> list[list]:
    # delete outer lines
    text = text[200:-400]

    # skip if file is too short
    if text == []:
        return None
    
    cleaned_text = []

    for line in text:
        # remove whitespace characters
        line.strip()

        # replace non-standard quotation mark
        line = re.sub(r'[\u2019]', "'", line)

        # replace characters - _ "" [] () and other quotes with spaces
        line = re.sub(r'[-_"\u2018\u201C\u201D\[\]\(\)]', " ", line)

        cleaned_text.append(line)
    
    # join the text as one string
    text = " ".join(cleaned_text)

    # split the text by punctuation
    text = re.split("[.;!?:]", text)

    cleaned_text = []

    for line in text:
        # require lines to be longer than two characters
        # discard all lines with non-alphabet characters
        if re.match(r'[a-zA-Z\s\'\,]+?$', line) and len(line)>2:
    
            # remove whitespace inside/outside/before commas
            line = line.strip()
            line = re.sub(r'\s+', " ", line)
            line = re.sub(r'\s+,', " ", line)

            cleaned_text.append(line)
        else:
            continue

    return cleaned_text


# convert line to ipa
def convert_ipa(line:str, key_dict:dict) -> str:

    # convert to ipa
    line = ipa.convert(line)

    # discard lines that cannot be converted
    if "*" in line:
        return None

    # replace '
    line = line.replace(r"'", "")
    
    # discard lines that are too short
    length = len(line)
    if length <= 2:
        return None

    # convert to numbers
    converted_line = []
    i = 0 # position tracker

    while i < length:
        # check for double phonemes
        if line[i:i+2] in key_dict:
            converted_line.append(key_dict[line[i:i+2]])
            i += 2
        else:
            converted_line.append(key_dict[line[i]])
            i += 1
    return converted_line


# main preparation function
def prepare(paths:list, num:int):
    
    # randomly select which texts to convert
    paths = rand_text(paths, num)

    for file_name in tqdm(paths):
        # get file key
        # MODIFICATION FOR WINDOWS
        # file_key = re.match(r'.*\\(\d+)', file_name).group(1)
        file_key = re.match(r'.*/(\d+)', file_name).group(1)

        # new file name
        new_file_name = file_key + ".gz"
        
        # check to see if the cleaned file already exists
        if Path(new_file_name).is_file():
            continue
    
        else:
            # get the encoding
            enc = keys_list[file_key][1]

            # open files with encoding
            try:
                with open(file_name, encoding=enc, newline="") as raw_file:
                    raw_text = [line for line in raw_file]

            # some files have continuation bytes in Latin-1 encoding
            except UnicodeDecodeError:
                enc = "latin-1"
                with open(file_name, encoding=enc, newline="") as raw_file:
                    raw_text = [line for line in raw_file]

            # clean up using string methods
            clean_text = str_clean_phase_1(raw_text)

            # move on if file is too short
            if clean_text == None:
                continue
            
            # convert to IPA using parallelization
            with concurrent.futures.ProcessPoolExecutor() as ppe:
                futures = [ppe.submit(convert_ipa, row, ipa_num) for row in clean_text]
                converted = [future.result() for future in concurrent.futures.as_completed(futures) if future.result() is not None]

            # write data to file
            with gzip.open(new_file_name, 'wt', encoding='utf-8') as f:
                json.dump(converted, f)


if __name__ == "__main__":

    # import dictionaries
    with open("links_list.json", "r") as key_file:
        keys_list = json.load(key_file)

    with open("ipa_num.json", "r") as dict_file:
        ipa_num = json.load(dict_file)

    # change directory
    os.chdir(Path("clean_texts"))

    # grab list of paths
    path_name = "../../get_resources/raw_texts"
    path_list = [str(path) for path in Path(path_name).rglob("*.txt")]
    
    # preparation parameters
    n_files = 1000
    n_keys = len(keys_list)
    
    # prepare the data
    # optionally, use n_keys to convert all the books
    prepare(path_list, n_files)
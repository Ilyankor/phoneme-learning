import os
import re
import json
from bs4 import BeautifulSoup

# make a dictionary
def make_dict() -> dict:
    # open the concatenated html
    with open("raw_list.txt", "r") as html_file:
        soup = BeautifulSoup(html_file, features="html.parser")
    
    file_dict = {}
    enc_dict = {"ascii": 0, "utf-8-sig": 1, "latin-1": 2}

    for link in soup.findAll("a"):

        # parse the link
        link = link.get("href")

        # eliminate non-zip links
        if link[-3:] != "zip":
            continue

        # get the key
        file_key = (re.search(r"/(\d+)(?=/[^/]+.zip$)", link)).group(1)

        # get the encoding
        if link[-6:-4] == "-0":
            enc = "utf-8-sig"
        elif link[-6:-4] == "-8":
            enc = "latin-1"
        else:
            enc = "ascii"
        
        # check against the dictionary
        if file_key in file_dict:
            _, enc0 = file_dict[file_key]

            if enc_dict[enc0] < enc_dict[enc]:
                continue
            else:
                file_dict[file_key] = (link, enc)
        else:
            file_dict[file_key] = (link, enc)
    
    # save the dictionary
    with open("links_list.json", "w") as f:
        json.dump(file_dict, f)

    return file_dict

# make list of links
def make_list(file_list:dict):

    links_list = [val[0]+"\n" for val in file_list.values()]

    with open("list.txt", "w") as f:
        f.writelines(links_list)

if __name__ == "__main__":
    # change directory
    os.chdir("raw_texts")

    # get list of links
    link_file = make_dict()

    # make the list as a text file
    make_list(link_file)
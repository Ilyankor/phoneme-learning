
# Phoneme text generation

## About

A project exploring the use of machine learning for text generation.
The text generated was used for a song.
Originally done in September of 2022 and primarily implemented in MATLAB, however, all files were accidentally lost.
This is a recreation and slight upgrade in Python.

## Steps

A good Python version to have is `3.11`.
The required packages are provided in `requirements.txt`.

### Get the data

1. In the Terminal, run

    ```bash
    cd get_resources
    chmod +x get_links.sh
    ./get_links.sh
    ```

   This gets the links to all the English books in the [Gutenberg Project](https://www.gutenberg.org/) and cleans up the extra folders and files.
   Note that `wget` is required.
   This will take about 16 minutes.

2. Still within the `get_resources` directory, run `make_list.py`.
   This creates a dictionary of the books `links_list.json` as well as list of all the links `list.txt`.

3. In the Terminal, in `get_resources`, run `download.sh`.
   This uses `wget` to download all the `.zip` files from the list of links provided by `list.txt`.
   The results are stored in the same folder, that is, `get_resources/raw_texts`.
   This will take quite a long time, about 2 hours and 40 minutes, if given a decent connection to the Internet.

   In addition, after having downloaded of all the files, it unzips all the files and cleans up the `raw_texts` folder.

    ```bash
    chmod +x download.sh
    ./download.sh
    ```

### Preprocess the data

The `eng_to_ipa` package is required for the conversion between English and phonemes.
It is not available via `pip` and must be installed manually.
[Here is a link.](https://github.com/mphilli/English-to-IPA)

1. In the `prepare_data` folder, run `prepare_data.py`.
   This script cleans each file in preparation for training.
   Several steps are taken to ensure quality data:
   - Trim the file to remove the Gutenberg information.
   - Remove non-alphabet characters and the lines that contain them.
   - Remove any extra white spaces.
   - Record the texts as a `list`, with each nested `list` the length of a full phrase.

   The lines are then converted into IPA, and then from IPA into integers.
   The conversion from IPA to integers is outlined in `ipa_num.json`.

   Note: the conversion process takes a **VERY LONG TIME**.
   Given a decent computer, this is estimated to take upwards of 100 hours.
   If all the books (as of May 2024, around 50,000) cannot be converted in a timely manner, the `prepare` function has a parameter `num` for which the number of files desired can be chosen at random and converted.
   Additionally, the script can be run multiple times to get more books converted (duplicates are handled appropriately).

   The cleaned files are found in `prepare_data/clean_texts`.

2. Run `split_data.py` in the same directory, `prepare_data`.
   This step transforms the variable length data in `/clean_texts` into arrays with fixed sequence lengths (defined by the parameter `seq_length`) in preparation for feeding into [PyTorch](https://pytorch.org/).
   It also splits the data into a training set and a testing set with ratios that are set by the `ratio` variable.
   The variable should be the fraction of the text that the test data should have.
   The data is stored using the [`zarr`](https://zarr.readthedocs.io/en/stable/index.html) format in the folder `learning/data`.

   Again, the script can be run multiple times in the event more books are converted.

3. To ensure that duplicate rows are discarded, run `remove_duplicates.py`.
   The script may require administrator privileges.

### Training the model

The machine learning model used is an LSTM.
The code for using PyTorch to train this model is largely adapted from [this article](https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/).

The model is defined in `learning/phoneme_model.py` and its parameters can be altered.

1. In the main directory `phoneme-learning`, run `training.py`.
   The training parameters such as number of epochs to run and the batch size can be modified in the `main` function of this file.

   Since the dataset is large, an additional parameter is provided to select chunks of `load_size` number of rows at a time instead of all at once in memory.
   The chunks are randomly selected and then fed into the `Dataloader` for batch training.

   After each epoch, a checkpoint is saved in `learning/model_weights.tar`.
   This script can eaily be run over multiple sessions, since it allows training to resume from the information given in the checkpoint file.

2. To generate text, run `generate.py`.
   The output is shown in `results.txt`.

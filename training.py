import os
import zarr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from pathlib import Path
from learning.phoneme_model import PhonemeModel


def main():
    # change directory
    os.chdir("learning")
        
    # training parameters
    num_epochs = 1 # 10
    batch_size = 256 # 4096
    model = PhonemeModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimiser and loss functions
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss(reduction="sum")

    # load previous state if available and initialise loss
    weights_path = Path("model_weights.tar")

    if weights_path.exists():
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
        current_loss = checkpoint['loss']

    else:
        current_model = None
        current_epoch = 0
        current_loss = np.inf

    # number of sequences to load into the memory at once
    load_size = 2048 # 524288 

    # open the files to see number of rows
    train_data = zarr.open(Path("data/train_data_clean.zarr"), "r")
    test_data = zarr.open(Path("data/test_data_clean.zarr"), "r")
    M = train_data.shape[0]
    N, seq_length = test_data.shape
    seq_length -= 1

    # calculate the number of batches for each epoch
    I = int(np.ceil(M/load_size))
    J = int(np.ceil(N/load_size))

    # random generator for random selections
    rng = np.random.default_rng()

    # train and test the model
    for epoch in tqdm(range(current_epoch+1, current_epoch+num_epochs+1)):
        # training mode
        model.train() 

        # permute the indices
        train_indices = rng.permutation(M)

        # batch training
        for i in tqdm(range(I)):
            rng_indices = train_indices[i*load_size:((i+1)*load_size)]

            # import data in chunks of load_size
            seq = train_data[rng_indices]
            X = seq[:, :-1]
            y = seq[:, -1]
            
            # convert to torch tensors
            num_seq = X.shape[0]

            X = torch.tensor(X, dtype=torch.float32).reshape(num_seq, seq_length, 1)
            X = X / float(48)
            y = torch.tensor(y, dtype=torch.int64)

            # use torch DataLoader to shuffle data and train
            loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)
            for X_batch, y_batch in loader:
                y_pred = model(X_batch.to(device))
                loss = loss_fn(y_pred, y_batch.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluation mode
        model.eval()
        loss = 0

        # permute the indices
        test_indices = rng.permutation(N)
        
        # validate the model
        for j in tqdm(range(J)):

            rng_indices = test_indices[j*load_size:((j+1)*load_size)]

            # import data in chunks of load_size
            seq = test_data[rng_indices]
            X = seq[:, :-1]
            y = seq[:, -1]
            
            # convert to torch tensors
            num_seq = X.shape[0]

            X = torch.tensor(X, dtype=torch.float32).reshape(num_seq, seq_length, 1)
            X = X / float(48)
            y = torch.tensor(y, dtype=torch.int64)

            # use torch DataLoader to shuffle data and evaluate
            loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    y_pred = model(X_batch.to(device))
                    loss += loss_fn(y_pred, y_batch.to(device))

        # compute the loss and update model if it is better
        if loss < current_loss:
            current_loss = loss
            current_model = model.state_dict()
        
        # print information out
        print("Epoch %d: Loss: %.4f" % (epoch, loss))
    
        # save a checkpoint
        torch.save({
                "epoch": epoch,
                "model_state_dict": current_model,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": current_loss}, 
                "model_weights.tar")


if __name__ == "__main__":
    main()
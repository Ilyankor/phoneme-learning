import zarr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from pathlib import Path
from phoneme_design import PhonemeModel


def main():
    # training parameters
    batch_size = 100 # PARAMETER
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

        current_model = model.state_dict()
        current_opt = optimizer.state_dict()
        current_epoch = checkpoint['epoch']
        current_loss = checkpoint['loss']

    else:
        current_model = None
        current_epoch = 0
        current_loss = np.inf

    # open the files and extract information
    train_data = zarr.open(Path("../../data/design.zarr"), "r")
    test_data = zarr.open(Path("../../data/test.zarr"), "r")
    M = train_data.cdata_shape[0]
    N = test_data.cdata_shape[0]
    _, seq_length = test_data.shape
    seq_length -= 1

    # random generator for random selections
    rng = np.random.default_rng()

    # train and test the model
    # permute the chunks
    train_indices = rng.permutation(M)

    # chunk training
    for i in tqdm(range(M)):
        # training mode
        model.train() 

        # import data
        seq = train_data.get_block_selection(train_indices[i])
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
    
        # validate the model
        # evaluation mode
        model.eval()
        loss = 0

        # random index
        j = rng.choice(N)

        # import data
        seq = test_data.get_block_selection(j)
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
            current_opt = optimizer.state_dict()
        
        # print information out
        print("Epoch %d: Loss: %.4f" % (i, loss))

        with open("log.txt", "a") as log_file:
            log_file.write(f"{i}, {loss} \n")

        # save a checkpoint
        torch.save({
                "epoch": i,
                "model_state_dict": current_model,
                "optimizer_state_dict": current_opt,
                "loss": current_loss}, 
                "model_weights.tar")


if __name__ == "__main__":
    main()

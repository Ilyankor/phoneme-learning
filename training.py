import zarr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from pathlib import Path
from phoneme_model import PhonemeModel


def main():
    # training parameters
    num_epochs = 10
    batch_size = 4096

    # model information
    model = PhonemeModel()
    num_char = 49

    # move the model to the gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # optimiser and scheduler (steps_per_epoch is defined by the number of chunks in the training data)
    optimizer = optim.AdamW(model.parameters(), lr=0.004)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1124, epochs=num_epochs)

    # loss function
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    
    # load previous state if available and initialise variables
    weights_path = Path("model_weights.tar")

    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        current_model = model.state_dict()
        current_opt = optimizer.state_dict()
        current_epoch = checkpoint["epoch"]
        current_loss = checkpoint["loss"]

    else:
        current_model = None
        current_epoch = 0
        current_loss = np.inf

    # open the data files and extract information
    train_data = zarr.open(Path("data/train.zarr"), "r")
    test_data = zarr.open(Path("data/test.zarr"), "r")
    M = train_data.cdata_shape[0]
    N = test_data.cdata_shape[0]
    _, seq_length = test_data.shape
    seq_length -= 1

    # random generator for random selections
    rng = np.random.default_rng()

    # turn on cudNN benchmarking
    torch.backends.cudnn.benchmark = True

    # enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # train and test the model
    for epoch in tqdm(range(current_epoch+1, current_epoch+num_epochs+1)):
        # training mode
        model.train() 

        # permute the chunks
        train_indices = rng.permutation(M)

        # chunk training
        for i in tqdm(range(M)):

            # import data
            seq = train_data.get_block_selection(train_indices[i])
            X = seq[:, :-1]
            y = seq[:, -1]
            
            # convert to torch tensors and normalize input
            num_seq = X.shape[0]

            X = torch.as_tensor(X, dtype=torch.float32).reshape(num_seq, seq_length, 1)
            X = X / float(num_char)
            y = torch.as_tensor(y, dtype=torch.int64)

            # use torch DataLoader to shuffle data
            loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
            
            # training sequence
            for X_batch, y_batch in loader:
                # zero out the gradient
                optimizer.zero_grad()
                
                # using mixed precision, step through the training
                with torch.cuda.amp.autocast():
                    y_pred = model(X_batch.to(device, non_blocking=True))
                    loss = loss_fn(y_pred, y_batch.to(device, non_blocking=True))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # step the scheduler
            scheduler.step()
    
        # validate the model
        # evaluation mode
        model.eval()
        validation_loss = 0

        # permute the chunks
        test_indices = rng.permutation(N)
            
        # validate the model
        for j in tqdm(range(N)):

            # import data
            seq = test_data.get_block_selection(test_indices[j])
            X = seq[:, :-1]
            y = seq[:, -1]
            
            # convert to torch tensors and normalize input
            num_seq = X.shape[0]

            X = torch.as_tensor(X, dtype=torch.float32).reshape(num_seq, seq_length, 1)
            X = X / float(num_char)
            y = torch.as_tensor(y, dtype=torch.int64)

            # use torch DataLoader to shuffle data and evaluate
            loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
            with torch.no_grad():
                for X_batch, y_batch in loader:
                    y_pred = model(X_batch.to(device, non_blocking=True))
                    validation_loss += loss_fn(y_pred, y_batch.to(device, non_blocking=True))

        # compute the loss and update model if it is better
        if validation_loss < current_loss:
            current_loss = validation_loss
            current_model = model.state_dict()
            current_opt = optimizer.state_dict()
        
        # print information out
        print(f"Epoch: {epoch} Loss: {validation_loss}")

        # save information to a log file
        with open("log.txt", "a") as log_file:
            log_file.write(f"Epoch: {epoch}, Loss: {validation_loss} \n")

        # save a checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": current_model,
            "optimizer_state_dict": current_opt,
            "loss": current_loss}, 
            "model_weights.tar")


if __name__ == "__main__":
    main()
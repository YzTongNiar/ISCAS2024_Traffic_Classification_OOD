import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model
import dataset


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

def time_slice(encoding_data, time_len, ran_seed):
    np.random.seed(ran_seed)
    time_sliced_data1 = []
    series_length = []
    for i in encoding_data:
        ifeature = i[0]
        time_max_index = np.argmin(abs(ifeature.iloc[:, 0].to_numpy() - (ifeature.iloc[-1, 0] - time_len)))
        if time_max_index == 0:
            index1 = 0
        else:
            index1 = np.random.randint(time_max_index)
        index2 = np.argmin(abs(ifeature.iloc[:, 0].to_numpy() - (ifeature.iloc[index1, 0] + time_len)))
        sliced_feature = ifeature.iloc[index1:index2+1, :]
        series_length.append(len(sliced_feature))
        time_sliced_data1.append((sliced_feature, i[1]))
    return time_sliced_data1, max(series_length)


def dataset_slice(time_sliced_data):
    train_set = []
    index = []
    test_set = []
    for i in range(5):
        for j in range(80):
            train_set.append(time_sliced_data[i*100+j])
            index.append(i*100+j)

    test_set_index = [i for i in range(800) if i not in index]
    for i in test_set_index:
        test_set.append(time_sliced_data[i])

    return train_set, test_set


def model_training(encoding_data, training_para, slice_time=1000, randomseed=30):

    time_sliced_data, maxlength = time_slice(encoding_data, slice_time, randomseed)

    train_dataset, test_dataset = dataset_slice(time_sliced_data)

    with open("Train_dataset.pickle", "wb") as f:
        # Serialize the data and write it to the file
        pickle.dump(train_dataset, f)
    with open("Test_dataset.pickle", "wb") as f:
        # Serialize the data and write it to the file
        pickle.dump(test_dataset, f)

    lr = training_para['Learning_rate']
    n_epochs = training_para['epcochs']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    np.random.seed(randomseed)
    torch.manual_seed(randomseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    Lstm_model = model.LSTMClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(Lstm_model.parameters(), lr=lr)

    train_sequences, test_sequences = train_test_split(train_dataset, test_size=0.2, random_state=2)
    print('START2')

    trainset = dataset.AppsDataset(train_sequences)
    valset = dataset.AppsDataset(test_sequences)

    batch_size = training_para['Batch_size']
    trainloader = DataLoader(trainset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)
    valloader = DataLoader(valset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)

    print('Start model training')

    best_acc = 0
    best_acc_train = 0
    patience = training_para['patience']
    patience_counter = 0
    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()



    #
    for epoch in range(1, n_epochs + 1):
        train_data_feature = []
        train_data_label = []
        # initialize
        # losses
        loss_train_total = 0
        loss_val_total = 0

        # Training loop
        correct_train, total_train = 0, 0
        for i, batch_data in enumerate(trainloader):
            Lstm_model.train()
            x_batch = batch_data['sequences'].to(device)

            # noise = torch.randn_like(x_batch.data)*3
            # x_batch = torch.nn.utils.rnn.PackedSequence(x_batch.data  + noise, x_batch.batch_sizes)

            y_batch = batch_data['label'].to(device).long()
            X_seq_len = batch_data['seq_len']

            y_pred, lstm_out_forward = Lstm_model(x_batch, X_seq_len)
            y_pred_soft = F.softmax(y_pred, dim=1)
            loss = criterion(y_pred_soft, y_batch)


            class_predictions = F.softmax(y_pred, dim=1).argmax(dim=1)
            total_train += y_batch.size(0)
            correct_train += (class_predictions == y_batch).sum().item()

            loss_train_total += loss.cpu().detach().item() * batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_data_feature.append(to_np(lstm_out_forward))
            train_data_label.append(to_np(y_batch))


        loss_train_total = loss_train_total / len(trainset)
        acc_train = correct_train / total_train



        # Validation loop
        correct, total = 0, 0
        with torch.no_grad():
            for i, batch_data in enumerate(valloader):
                Lstm_model.eval()
                x_batch = batch_data['sequences'].to(device)
                y_batch = batch_data['label'].to(device).long()
                X_seq_len = batch_data['seq_len']
                y_pred, lstm_out_forward_val = Lstm_model(x_batch, X_seq_len)
                loss = criterion(y_pred, y_batch)
                loss_val_total += loss.cpu().detach().item() * batch_size

                class_predictions = F.softmax(y_pred, dim=1).argmax(dim=1)
                total += y_batch.size(0)
                correct += (class_predictions == y_batch).sum().item()

        loss_val_total = loss_val_total / len(valset)
        acc = correct / total


        # Logging
        if epoch % 1 == 0:
            print(
                f'Epoch: {epoch:3d}. Train Loss: {loss_train_total:.4f}. '
                f'Traing_Acc: {acc_train: 2.2%}'
                f'Val Loss: {loss_val_total:.4f} Acc.: {acc:2.2%}')

        if acc >= best_acc:
            patience_counter = 0
            best_acc = acc
            saved_dict = {
                'model': Lstm_model.state_dict(),
                'opt': optimizer.state_dict()
            }

            torch.save(saved_dict, '.best_model.pth')
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
            with open("Train_data_feature.pickle", "wb") as f:
                # Serialize the data and write it to the file
                train_data_feature = concat(train_data_feature).copy()
                pickle.dump(train_data_feature, f)
            with open("Train_data_label.pickle", "wb") as f:
                # Serialize the data and write it to the file
                train_data_label = concat(train_data_label).copy()
                pickle.dump(train_data_label, f)

        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
        if acc_train > best_acc_train:
            best_acc_train = acc_train
    return best_acc_train, best_acc







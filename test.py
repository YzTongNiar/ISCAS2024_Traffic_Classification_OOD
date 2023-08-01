import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sklearn.metrics as sk

from numpy.linalg import norm, pinv
import model
import dataset
from sklearn.covariance import EmpiricalCovariance

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle

def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh
def fpr_recall(ind_conf, ood_conf, tpr = 0.95):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr
def load_feature():
    with open("Train_data_feature.pickle", "rb") as f:
        # Deserialize the data and load it into memory
        train_data_feature = pickle.load(f)
    return train_data_feature


def load_index(indexname):

    if indexname==0:
        test_index = np.concatenate((np.array([[i for i in range(100)]]),
                                      np.array([[i for i in range(140, 160)],
                                                [i for i in range(160, 180)],
                                                [i for i in range(180, 200)],
                                                [i for i in range(280, 300)],
                                                [i for i in range(300, 320)]])), axis=None)

    elif indexname==1:
        test_index = np.concatenate((np.array([[i for i in range(100)]]),
                                      np.array([[i for i in range(100, 120)],
                                                [i for i in range(120, 140)],
                                                [i for i in range(200, 220)],
                                                [i for i in range(220, 240)],
                                                [i for i in range(260, 280)]])), axis=None)

    elif indexname==2:
        test_index = np.concatenate((np.array([[i for i in range(100)]]),
                                      np.array([[i for i in range(240, 260)],
                                                [i for i in range(320, 340)],
                                                [i for i in range(340, 360)],
                                                [i for i in range(360, 380)],
                                                [i for i in range(380, 400)]])), axis=None)
    return test_index



def classfy_acc(test_dataset, training_para, Lstm_model, device):

    valset = dataset.AppsDataset(test_dataset)
    batch_size = training_para['Batch_size']
    valloader = DataLoader(valset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)

    correct, total = 0, 0
    with torch.no_grad():
        for i, batch_data in enumerate(valloader):
            Lstm_model.eval()
            x_batch = batch_data['sequences'].to(device)
            y_batch = batch_data['label'].to(device).long()
            X_seq_len = batch_data['seq_len']
            y_pred, lstm_out_forward_val = Lstm_model(x_batch, X_seq_len)
            class_predictions = F.softmax(y_pred, dim=1).argmax(dim=1)
            total += y_batch.size(0)
            correct += (class_predictions == y_batch).sum().item()

    acc = correct / total

    return acc


def model_val(encoding_data, training_para, slice_time_list, pca_acum_engi):
    train_id_feature = load_feature()



    with open("Train_data_label.pickle", "rb") as f:
        # Deserialize the data and load it into memory
        train_data_label = pickle.load(f)


    with open("Test_dataset.pickle", "rb") as f:
        # Deserialize the data and load it into memory
        test_dataset = pickle.load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Lstm_model = model.LSTMClassifier().to(device)
    checkpoint = torch.load('.best_model.pth')
    Lstm_model.load_state_dict(checkpoint['model'])

    test_acc = classfy_acc(test_dataset[:100], training_para, Lstm_model, device)

    print(test_acc)




    for i, idata in enumerate(test_dataset):
        idata = list(idata)
        if i < 100:
            idata[1] = 1
        else:
            idata[1] = 0
        test_dataset[i] = tuple(idata)

    valset = dataset.AppsDataset(test_dataset)
    batch_size = training_para['Batch_size']
    valloader = DataLoader(valset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate_fn)

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()

    _score_energy = []
    _score_softmax = []
    _label = []
    test_data_feature = []
    logit = []


    with torch.no_grad():
        for i, batch_data in enumerate(valloader):
            Lstm_model.eval()
            X_batch = batch_data['sequences'].to(device)
            y_batch = batch_data['label'].to(device).long()
            X_seq_len = batch_data['seq_len']
            y_pred, lstm_out = Lstm_model(X_batch, X_seq_len)

            _score_energy.append(-to_np((1 * torch.logsumexp(y_pred / 1, dim=1))))
            _score_softmax.append(-np.max(to_np(F.softmax(y_pred, dim=1)), axis=1))
            _label.append(to_np(y_batch))
            test_data_feature.append(to_np(lstm_out))
            # logit.append((to_np(F.softmax(y_pred, dim=1))))
            logit.append(to_np(y_pred))

    test_data_feature = concat(test_data_feature).copy()
    logit = concat(logit).copy()
    _score_softmax = concat(_score_softmax).copy() #MSP
    _score_energy = concat(_score_energy).copy()   #Energy
    _label = concat(_label).copy()


    '''
    VIM score
    '''
    w = Lstm_model.fc.weight.cpu().detach().squeeze().numpy()
    b = Lstm_model.fc.bias.cpu().detach().squeeze().numpy()
    u = -np.matmul(pinv(w), b)
    print('computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    # ec.fit(train_id_feature - u)
    ec.fit(train_id_feature)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    acuumulate = 0
    dim = 1
    for i, values in enumerate (eig_vals):
        acuumulate += values/ np.sum(eig_vals)
        dim +=1
        if acuumulate > pca_acum_engi:
            break
    if acuumulate == 0:
        dim = 0
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[dim:]]).T)
    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(train_id_feature - u, NS), axis=-1)
    logit_id_train = train_id_feature @ w.T + b
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
    print(f'{alpha=:.4f}')
    # vim_score = norm(np.matmul(test_data_feature - u, NS), axis=-1) * alpha

    vim_score = norm(np.matmul(test_data_feature , NS), axis=-1)



    metrics = []
    for i in range(3):
        test_index_ = load_index(i)
        test_label = [_label[i] for i in test_index_]
        test_energy = [_score_energy[i] for i in test_index_]
        test_softmax = [_score_softmax[i] for i in test_index_]
        test_vim_score = [vim_score[i] for i in test_index_]

        auroc_softmax = sk.roc_auc_score(np.array(test_label), -np.array(test_softmax))
        aupr_softmax = sk.average_precision_score(np.array(test_label), -np.array(test_softmax))

        auroc_energy = sk.roc_auc_score(np.array(test_label), -np.array(test_energy))
        aupr_energy = sk.average_precision_score(np.array(test_label), -np.array(test_energy))

        auroc_vim= sk.roc_auc_score(np.array(test_label), -np.array(test_vim_score))
        aupr_vim = sk.average_precision_score(np.array(test_label), -np.array(test_vim_score))


        fpr_energy= fpr_recall(-np.array(test_energy)[:100], -np.array(test_energy)[100:])
        fpr_softmax = fpr_recall(-np.array(test_softmax)[:100], -np.array(test_softmax)[100:])
        fpr_vim = fpr_recall(-np.array(test_vim_score)[:100], -np.array(test_vim_score)[100:])


        print(fpr_recall(-np.array(test_energy)[:100], -np.array(test_energy)[100:]))

        metrics.append([[auroc_softmax, auroc_energy, auroc_vim],
                       [aupr_softmax, aupr_energy, aupr_vim],
                        [fpr_softmax, fpr_energy, fpr_vim]])


    return metrics

import sys
from model import *
from utils import *
from evalution import *
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
def ViewContrastiveLoss(view_i, view_j, batch,temperature):

    z_i = F.normalize(view_i, dim=1)
    z_j = F.normalize(view_j, dim=1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                            dim=2)
    similarity_matrix = similarity_matrix.to(device)
    sim_ij = torch.diag(similarity_matrix, batch)
    sim_ji = torch.diag(similarity_matrix, -batch)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    negatives_mask = torch.ones(2 * batch, 2 * batch) - torch.eye(2 * batch, 2 * batch)
    negatives_mask = negatives_mask.to(device)
    denominator = negatives_mask * torch.exp(similarity_matrix / temperature)

    loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
    loss = torch.sum(loss_partial) / (2 * batch)

    return loss
def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, fold):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        n = data.y.shape[0]  # batch
        optimizer.zero_grad()
        output,x_g,hx_g= model(data,data.x,data.edge_index,data.batch,data.hedge_index,n,data.fp)
        loss_1 = criterion(output, data.y)
        T = 0.2
        loss_2 = ViewContrastiveLoss (x_g,hx_g,n,T)
        loss = loss_1 + 0.1*loss_2
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train fold:{} epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(fold, epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            n = data.y.shape[0]
            output,x_g,hx_g = model(data, data.x, data.edge_index, data.batch, data.hedge_index,n,data.fp)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.cpu()), 0)
    return total_labels,total_preds

if __name__ == "__main__":
    cuda_name = "cuda"
    if len(sys.argv) > 3:
        cuda_name = "cuda:" + str(int(sys.argv[3]))
    print('cuda_name:', cuda_name)

    TRAIN_BATCH_SIZE = 256
    TEST_BATCH_SIZE = 256
    LR = 0.0005
    LOG_INTERVAL = 20
    NUM_EPOCHS = 200
    set_random_seed(42)
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    fold_res = []
    for i in range(1,2):
        processed_train = f'data_HLM/fold-{i}/processed/' + 'train.pt'
        processed_test = f'data_HLM/fold-{i}/processed/' + 'test.pt'
        if ((not os.path.isfile(processed_train)) or (not os.path.isfile(processed_test))):
                print('please run create_data.py to prepare data in pytorch format!')
        else:
            train_data = TestbedDataset(root=f'data_HLM/fold-{i}', dataset='train')
            test_data = TestbedDataset(root=f'data_HLM/fold-{i}', dataset='test')

            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = HyperGF().to(device)
            criterion = nn.BCEWithLogitsLoss()
            contrastive_loss = nn.CrossEntropyLoss(reduction='mean')
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            max_auc = 0

            model_file_name = 'bestmodel' + '.pt'
            result_file_name = 'result' + '.csv'
            for epoch in range(NUM_EPOCHS):
                train(model, device, train_loader, optimizer, epoch + 1, i)
                G, P = predicting(model, device, test_loader)

                auc, acc, precision, recall, f1_scroe, mcc = metric(G, P)
                ret = [auc, acc, precision, recall, f1_scroe, mcc]
                if auc > max_auc:
                    max_auc = auc
                    cur_res = ret
                    torch.save(model.state_dict(), model_file_name)
                    with open(result_file_name, 'w') as f:
                        f.write(','.join(map(str, ret)))
                print('%.4f\t %.4f\t %.4f\t %.4f\t%.4f\t %.4f' % (auc, acc, precision, recall, f1_scroe, mcc))

            print('Maximum acc found. Model saved.')
        fold_res.append(cur_res)
        print(fold_res)
        pd.DataFrame(fold_res,columns=['auc', 'acc', 'precision', 'recall', 'f1_scroe', 'mcc']).to_csv('./10fold.csv')

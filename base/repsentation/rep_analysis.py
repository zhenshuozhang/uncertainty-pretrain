import dgl
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t()).mean()

def cos_sim_matrix(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def l2_sim_matrix(z1, z2):
    l2_dis = lambda x, y: torch.sqrt(torch.mm(x-y, (x-y).t()))
    n = z2.shape[0]
    res = []
    for i in range(n):
        dis = torch.cdist(z1, z2[i].unsqueeze(0))
        res.append(dis)

    res = torch.cat(res, dim=-1)
    return res


def get_all_rep(model, loader, device, num_tasks):
    model.eval()
    y_rep = []
    y_true = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        print(batch)
        batch = batch.to(device)

        with torch.no_grad():
            #rep = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            rep = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(-1, num_tasks))
        y_rep.append(rep)

    y_true = torch.cat(y_true, dim = 0)
    y_rep = torch.cat(y_rep, dim = 0)

    return  y_true, y_rep

def encoder_snr(model, loader, device, args):
    model.eval()
    l2_dis = lambda x, y: torch.sqrt(torch.mm(x-y, (x-y).t()))
    y_true, y_rep = get_all_rep(model, loader, device, args.num_tasks)
    n_sample = y_true.shape[0]
    dim = y_rep.shape[1]
    n_classes = 2
    
    score = []

    for _ in range(y_true.shape[-1]):
        samples_in_label = [[] for i in range(n_classes)]
        labels = ((y_true[:, _] + 1)/2).to(torch.int)
        
        # record sample id indexed by label
        for i in range(n_sample):
            samples_in_label[labels[i]].append(i)
        # convert to long tensor
        for i in range(n_classes):
            samples_in_label[i] = torch.tensor(samples_in_label[i]).to(torch.long)
        # calculate mean feat of every label
        mean_feat_in_label = torch.zeros(n_classes, dim)
        for i in range(n_classes):
            mean_feat_in_label[i] = y_rep[samples_in_label[i]].mean(dim=0) # record mean feat for every label
        mean_feat = torch.Tensor(mean_feat_in_label).to(device)
        # prepare to calculate snr
        snr_in_label = torch.zeros(n_classes)
        s_in_label = [torch.Tensor([]).to(device) for i in range(n_classes)]
        n_in_label = [torch.Tensor([]).to(device) for i in range(n_classes)]

        sim_matrix = l2_sim_matrix(y_rep, mean_feat)
        #sim_matrix = cos_sim_matrix(y_rep, mean_feat)

        # delete the nan values
        nan_idx = []
        for i in range(n_classes):
            if torch.isnan(sim_matrix[0][i]):
                nan_idx.append(i)
        mask = torch.ones(sim_matrix.shape[1], dtype=torch.bool)
        mask[nan_idx] = False
        reg_sim_matrix = sim_matrix[:, mask] 

        value = torch.zeros(n_sample)

        # calculate similarity for every node compare to every label
        for i in range(n_sample):
            value[i] = sim_matrix[i][labels[i]] / sim_matrix[i][abs(1-labels[i])]
            for j in range(n_classes):
                if j == labels[i]:
                    value[i] = sim_matrix[i][j]
                    s_in_label[j] = torch.cat((s_in_label[j], sim_matrix[i][j].unsqueeze(0)), dim=0)
                else:
                    n_in_label[j] = torch.cat((n_in_label[j], sim_matrix[i][j].unsqueeze(0)), dim=0)
        
        # calculate snr
        snr_in_label = torch.zeros(n_classes)
        for i in range(n_classes):
            snr_in_label[i] = s_in_label[i].mean() / n_in_label[i].mean()

        #score.append(F.cosine_similarity(mean_feat[0].unsqueeze(0), mean_feat[1].unsqueeze(0)))
        #score.append(l2_dis(mean_feat[0].unsqueeze(0), mean_feat[1].unsqueeze(0)).item())
        #score.append(snr_in_label)
        score.append(np.mean(value.numpy()))
    
    return score

def show_rep_snr(model, loader, device, args, tag=''):
    snr_in_label = encoder_snr(model, loader, device, args)
    print(np.mean(snr_in_label))
    file = './compare_log.log'
    with open(file, 'a+') as f:
        f.write(f' dataset: {args.dataset} pretrain: {args.input_model_file} tag: {tag}\n')
        f.write(str(np.mean(snr_in_label)))
        f.write('\n\n')
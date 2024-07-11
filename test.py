import torch
from torchmetrics.functional.classification import binary_calibration_error
import numpy as np
import math
import random
import torch.nn.functional as F

"""
def expected_calibration_error(samples, true_labels, M=3):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

   # keep confidences / predicted "probabilities" as they are
    confidences = samples
    # get binary class predictions from confidences
    predicted_label = (samples>0.5).astype(float)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece


p1 = torch.Tensor([0.1,0.2,0.8])
lbs1 = torch.Tensor([0,0,1])

ece = binary_calibration_error(p1, lbs1, n_bins=3).item()
print(ece)

p2 = 1-p1
lbs2 = 1-lbs1

ece = binary_calibration_error(p2, lbs2, n_bins=3).item()

print(ece)

print(expected_calibration_error(p1.numpy(), lbs1.numpy()))
print(expected_calibration_error(p2.numpy(), lbs2.numpy()))
"""
"""
vars_ = 0.2698
preds_ = -1488.8058
lbs_ = -1478.1555
n = (math.log(vars_)+((preds_ - lbs_)*(preds_ - lbs_))/vars_)/2
print(n)
"""
"""
import torch
from torch.distributions import Normal
normal_dist = Normal(torch.tensor([0.0]), torch.tensor([1.0])) # 创建一个均值为0，标准差为1的正态分布对象
value = torch.tensor([0.0]) # 创建一个值为0的张量
log_prob_value = normal_dist.log_prob(value) # 计算该值在正态分布下的对数概率
print(log_prob_value) # 输出tensor([-0.9189])
"""

"""
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN, AffinityPropagation, HDBSCAN
from sklearn.manifold import MDS

# 假设M是一个n*n的相似矩阵，n是样本数
# 假设k是指定的聚类类别数
n = 50
M = np.random.rand(n, n)
M = (M + M.T)/2
print(M)
# 创建一个SpectralClustering对象，指定参数affinity为"precomputed"，表示输入的是相似矩阵
sc = SpectralClustering(n_clusters=2, affinity="precomputed")
#clustering = AffinityPropagation(n_clusters=2,random_state=5, affinity='precomputed').fit(M)
#clustering = DBSCAN(eps=0.5, min_samples=2).fit(M)
mds = MDS()
mds.fit(M)
a = mds.embedding_
print(a)
"""
"""
emb = torch.Tensor([[1,2], [2,2], [2,1]])
center_point = torch.Tensor([[1,2]])

mask = torch.any(emb != center_point.repeat(emb.shape[0], 1), dim=1)
emb = emb[mask]

print(mask)
"""
"""
a = torch.Tensor([0,1,2,3,4,5,6,7,8])
indices = [1,3,5]
mask = torch.Tensor([False for i in range(len(a))])
mask[indices] = True
out_indices = a[(mask==0)].tolist()
print(mask==0)
print(a[indices])
print(a[out_indices])



s = 'models_graphcl/graphcl_sub_5_0.6.pth'
parts = s.rsplit('graphcl_sub_', 1)
print(parts[-1][:-4])
"""
"""
a = torch.Tensor([[1,2,3],[3,4,5],[5,6,7]])
b = torch.triu(a)

print(b)
"""

"""
subset_mmd = [3,6,8,9,1]
subsets_index = [[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]

positions = list(range(len(subsets_index)))
random.shuffle(positions)
print(positions)
random_subsets_index = [subsets_index[x] for x in positions]
select_index = sum(random_subsets_index[:3], [])
print(select_index)
subset_order = np.argsort(subset_mmd)
ordered_subsets_index = [subsets_index[x] for x in subset_order]
select_index = sum(ordered_subsets_index[-3:], [])
#random.shuffle(select_index)
print(subset_order)
print(select_index)

from datetime import datetime
 
# 获取当前日期和时间
now = datetime.now()
 
# 输出当前日期和时间
print("现在的日期和时间是:", now.strftime('%Y-%m-%d %H:%M:%S'))
"""

print(min(0,1))
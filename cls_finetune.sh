split=scaffold
pre_model=models_graphcl/graphcl_80.pth
ue_method=svdkl

#for ue_method in none focal dkl
#do
#for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
for dataset in bace bbbp clintox
do
echo $dataset
python finetune.py --split $split --gnn_type gin --dataset $dataset --csv --runs 5 --use_cfg
#done
done
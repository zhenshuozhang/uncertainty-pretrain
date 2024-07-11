#for dataset in bace bbbp clintox hiv muv sider tox21 toxcast
#do
#echo $dataset
echo $1
for dkl_s in s1 s2 s3 s4
do
python finetune.py --input_model_file models_graphcl/graphcl_80.pth --gnn_type gin --dataset $1 --ue_method dkl --dkl_s $dkl_s --csv --runs 5 --epochs 50
done
#done

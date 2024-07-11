dataset=$1

python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'none'
python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_sub_4_0.2.pth'
python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_sub_6_0.4.pth'
python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_sub_5_0.6.pth'
python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_sub_7_0.8.pth'
#python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_80.pth'

python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_20.pth'
python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_40.pth'
python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_60.pth'
python finetune.py --use_exp --dataset $dataset --save_kernel --input_model_file 'models_graphcl/graphcl_80.pth'
dis_type=$1
dataset=$2

python ana_kernel.py --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'none'
python ana_kernel.py --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_sub_4_0.2.pth'
python ana_kernel.py --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_sub_6_0.4.pth'
python ana_kernel.py --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_sub_5_0.6.pth'
python ana_kernel.py --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_sub_7_0.8.pth'
#python ana_kernel.py --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_80.pth'

python ana_kernel.py  --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_20.pth'
python ana_kernel.py  --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_40.pth'
python ana_kernel.py  --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_60.pth'
python ana_kernel.py  --use_exp --dis_type $dis_type --dataset $dataset --input_model_file 'models_graphcl/graphcl_80.pth'
export CUDA_VISIBLE_DEVICES=0,1,2,3 # use the first 4 GPUs
torchrun --master_port 10001 --nproc_per_node=4 main.py config/cifar10_r18_connr.yml
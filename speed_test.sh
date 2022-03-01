# export ONEFLOW_DEBUG_MODE=True
#/opt/nvidia/nsight-systems/2021.4.1/target-linux-x64/nsys profile --stats=true -o log/oneflow_4x4_pop.qdrep python oneflow/upcunet_v3.py  --pretrain False --fp16 True --real_data False --graph False --profile True



#/opt/nvidia/nsight-systems/2021.4.1/target-linux-x64/nsys profile --stats=true -o log/torch_4x4_fp32.qdrep python3 torch/upcunet_v3.py  --pretrain False --fp16 False --real_data False --profile True


python3 oneflow/upcunet_v3.py \
 --pretrain False --fp16 True \
 --real_data False \
 --graph True --batch_size 16 \
 --conv_cudnn_search False 


#python3 torch/upcunet_v3.py  --pretrain False --fp16 True --real_data False --batch_size 16

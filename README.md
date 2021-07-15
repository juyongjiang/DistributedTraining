# DistributedTraining
Tutorial: [Distributed Training On Multiple GPUs With PyTorch](https://juyongjiang.medium.com/distributed-training-on-multiple-gpus-e0ee9c3d0126)

`ddp.py` is the example code of how to use `nn.DistributedDataParalllel` to implement distributed training in multiple GPUs with PyTorch. 

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```

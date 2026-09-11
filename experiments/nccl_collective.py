"""Run with torchrun --nproc-per-node=2 experiments/nccl_collective.py."""
import json
import os
import statistics

import torch
import torch.distributed as dist


def main():
    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    try:
        rank,world=dist.get_rank(),dist.get_world_size()
        base=torch.full((1<<20,),float(rank+1),device=f'cuda:{local_rank}')
        x=torch.empty_like(base)
        times=[]
        for i in range(30):
            x.copy_(base);dist.barrier()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();dist.all_reduce(x);end.record();end.synchronize()
            if i>=10:times.append(start.elapsed_time(end))
        assert torch.allclose(x,torch.full_like(x,world*(world+1)/2))
        if rank==0:print(json.dumps({'world_size':world,'payload_bytes':base.numel()*base.element_size(),
            'median_allreduce_ms':statistics.median(times),'gpu':torch.cuda.get_device_name(local_rank),'correctness_passed':True}))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()

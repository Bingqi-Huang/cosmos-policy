#!/usr/bin/env python3
import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn


def init(backend):
    local_rank = int(os.environ["LOCAL_RANK"])
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend, init_method="env://")
    return local_rank, dist.get_rank(), dist.get_world_size()


def finish(backend):
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def allreduce(rank):
    x = torch.ones((1024, 1024), device="cuda", dtype=torch.float32) * (rank + 1)
    for _ in range(20):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    if rank == 0:
        print("allreduce ok", float(x.flatten()[0]))


def broadcast(rank, dtype, size_mb, iters):
    dtype = getattr(torch, dtype)
    elem_size = torch.empty((), dtype=dtype).element_size()
    n = max(1, size_mb * 1024 * 1024 // elem_size)
    x = torch.empty((n,), device="cuda", dtype=dtype)
    if rank == 0:
        x.fill_(1)
    else:
        x.fill_(0)
    for _ in range(iters):
        dist.broadcast(x, src=0)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"broadcast ok dtype={dtype} size_mb={size_mb} iters={iters}")


def ddp(rank):
    model = nn.Sequential(nn.Linear(4096, 4096, bias=False), nn.GELU(), nn.Linear(4096, 4096, bias=False)).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(3):
        x = torch.randn(8, 4096, device="cuda")
        y = model(x).sum()
        y.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if rank == 0:
        print("ddp ok")


def gather_object(rank, world):
    obj = {"rank": rank, "payload": list(range(8))}
    out = [None for _ in range(world)] if rank == 0 else None
    dist.gather_object(obj, out, dst=0)
    if rank == 0:
        print("gather_object ok", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["allreduce", "broadcast", "ddp", "gather_object"])
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--size-mb", type=int, default=250)
    parser.add_argument("--iters", type=int, default=4)
    args = parser.parse_args()

    _, rank, world = init(args.backend)
    if rank == 0:
        print(
            "world",
            world,
            "backend",
            args.backend,
            "torch",
            torch.__version__,
            "cuda",
            torch.version.cuda,
            "nccl",
            torch.cuda.nccl.version(),
        )
    if args.backend != "nccl" and args.mode != "gather_object":
        raise ValueError(f"{args.mode} uses CUDA tensors and requires --backend=nccl")
    if args.mode == "allreduce":
        allreduce(rank)
    elif args.mode == "broadcast":
        broadcast(rank, args.dtype, args.size_mb, args.iters)
    elif args.mode == "ddp":
        ddp(rank)
    elif args.mode == "gather_object":
        gather_object(rank, world)
    finish(args.backend)


if __name__ == "__main__":
    main()

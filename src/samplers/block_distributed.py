import math
import torch
from torch.utils.data import Sampler
import torch.distributed as dist

class ZarrBlockDistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, drop_last=True):
        
        # --- ROBUST DDP DETECTION ---
        if num_replicas is None:
            if not dist.is_available():
                num_replicas = 1
            else:
                if dist.is_initialized():
                    num_replicas = dist.get_world_size()
                else:
                    num_replicas = 1 # Fallback for local testing

        if rank is None:
            if not dist.is_available():
                rank = 0
            else:
                if dist.is_initialized():
                    rank = dist.get_rank()
                else:
                    rank = 0 # Fallback for local testing
                
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.epoch = 0
        
        total_len = len(dataset)
        self.num_raw_batches = total_len // batch_size
        
        # Enforce divisibility for Sync
        if self.drop_last:
            self.num_batches_total = (self.num_raw_batches // self.num_replicas) * self.num_replicas
            self.num_batches_per_replica = self.num_batches_total // self.num_replicas
        else:
            self.num_batches_total = self.num_raw_batches
            self.num_batches_per_replica = math.ceil(self.num_batches_total / self.num_replicas)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        batch_starts = list(range(0, len(self.dataset), self.batch_size))
        
        if len(batch_starts) > self.num_raw_batches:
             batch_starts = batch_starts[:self.num_raw_batches]
             
        if self.drop_last:
            batch_starts = batch_starts[:self.num_batches_total]
        
        indices = torch.randperm(len(batch_starts), generator=g).tolist()
        batch_starts = [batch_starts[i] for i in indices]

        my_batch_starts = batch_starts[self.rank : len(batch_starts) : self.num_replicas]
        
        for start in my_batch_starts:
            end = min(start + self.batch_size, len(self.dataset))
            yield list(range(start, end))

    def __len__(self):
        return self.num_batches_per_replica
    
    def set_epoch(self, epoch):
        self.epoch = epoch
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader

class SyncedGroupLoaderFinite:
    def __init__(self, loaders: Dict[str, DataLoader], weights: Dict[str, float]):
        self.loaders = loaders
        self.loader_keys = sorted(list(loaders.keys()))
        self.epoch = 0
        total = sum(weights[k] for k in self.loader_keys)
        self.probs = [weights[k]/total for k in self.loader_keys]
        self.total_batches = sum(len(l) for l in loaders.values())
        
    def __iter__(self):
        # Auto-increment epoch each time iteration starts
        current_epoch = self.epoch
        self.epoch += 1
        
        # Propagate to underlying samplers
        for l in self.loaders.values():
            if hasattr(l.sampler, 'set_epoch'):
                l.sampler.set_epoch(current_epoch)
        
        iterators = {k: iter(v) for k, v in self.loaders.items()}
        rng = np.random.default_rng(seed=42 + current_epoch)
        
        completed_loaders = set()
        
        while len(completed_loaders) < len(self.loader_keys):
            valid_indices = [i for i, k in enumerate(self.loader_keys) if k not in completed_loaders]
            if not valid_indices: break
            
            valid_probs = np.array([self.probs[i] for i in valid_indices])
            if valid_probs.sum() == 0: break
            valid_probs = valid_probs / valid_probs.sum()
            
            choice_idx = rng.choice(len(valid_indices), p=valid_probs)
            group_name = self.loader_keys[valid_indices[choice_idx]]
            
            try:
                batch = next(iterators[group_name])
                batch['_group_name'] = group_name
                yield batch
            except StopIteration:
                completed_loaders.add(group_name)
                continue

    def __len__(self):
        return self.total_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        for l in self.loaders.values():
            if hasattr(l.sampler, 'set_epoch'):
                l.sampler.set_epoch(epoch)


class SyncedGroupLoader:
    """
    Infinite-mixing loader.
    Guarantees strict DDP synchronization by cycling smaller datasets.
    """
    def __init__(self, loaders: Dict[str, DataLoader], weights: Dict[str, float], steps_per_epoch: int = None):
        self.loaders = loaders
        self.loader_keys = sorted(list(loaders.keys()))
        self.epoch = 0
        # Normalize weights
        total = sum(weights[k] for k in self.loader_keys)
        self.probs = [weights[k]/total for k in self.loader_keys]
        
        # Define Epoch Length
        # Default: Sum of all batches (approx 1 pass over all data)
        if steps_per_epoch is None:
            self.total_batches = sum(len(l) for l in loaders.values())
        else:
            self.total_batches = steps_per_epoch
            
    def __iter__(self):
        # Auto-increment epoch each time iteration starts
        # This ensures different shuffling each epoch without external set_epoch call
        current_epoch = self.epoch
        self.epoch += 1
        
        # Store for _repeat_loader to access when cycling
        self._base_epoch = current_epoch
        
        # 1. Create INFINITE iterators
        # When a loader runs out, it automatically restarts
        # (sampler epoch is set inside _repeat_loader for each cycle)
        iterators = {k: iter(self._repeat_loader(v)) for k, v in self.loaders.items()}
        
        # 2. Deterministic RNG
        rng = np.random.default_rng(seed=42 + current_epoch)
        
        # 3. Iterate for fixed number of steps
        for _ in range(self.total_batches):
            # Select Group
            choice_idx = rng.choice(len(self.loader_keys), p=self.probs)
            group_name = self.loader_keys[choice_idx]
            
            # Fetch Batch (Guaranteed to exist because iterators are infinite)
            batch = next(iterators[group_name])
            batch['_group_name'] = group_name
            yield batch

    def _repeat_loader(self, loader):
        """Yields batches forever, with fresh shuffle each cycle."""
        cycle = 0
        while True:
            # Update sampler epoch for each cycle to get fresh shuffle
            # Use bit shifting to avoid seed collisions between epochs
            # Upper 16 bits: base epoch, Lower 16 bits: cycle count
            if hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch((self._base_epoch << 16) | (cycle & 0xFFFF))
            
            yielded_any = False
            for batch in loader:
                yielded_any = True
                yield batch
            
            # Guard against empty loaders causing infinite loops
            if not yielded_any:
                raise RuntimeError(
                    f"DataLoader yielded 0 batches - cannot cycle. "
                    f"Ensure group has at least batch_size * world_size samples."
                )
            
            cycle += 1

    def __len__(self):
        return self.total_batches
    
    def set_epoch(self, epoch):
        self.epoch = epoch
        for l in self.loaders.values():
            if hasattr(l.sampler, 'set_epoch'):
                l.sampler.set_epoch(epoch)
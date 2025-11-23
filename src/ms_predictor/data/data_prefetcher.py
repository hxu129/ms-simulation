"""
Data prefetcher for overlapping data transfer with GPU computation.

This class enables asynchronous data loading by prefetching the next batch
while the current batch is being processed on GPU.
"""

import torch
from typing import Iterator, Dict


class DataPrefetcher:
    """
    Prefetch data to GPU asynchronously.
    
    This overlaps data transfer with GPU computation for significant speedup.
    Requires pin_memory=True in DataLoader.
    """
    
    def __init__(self, loader: Iterator, device: torch.device):
        """
        Initialize data prefetcher.
        
        Args:
            loader: DataLoader or iterable
            device: Target device (usually GPU)
        """
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.next_batch = None
        self.preload()
    
    def preload(self):
        """Preload the next batch asynchronously."""
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        
        # Transfer to GPU asynchronously
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = {
                    k: v.to(self.device, non_blocking=True) 
                    for k, v in self.next_batch.items()
                }
        else:
            # CPU fallback
            self.next_batch = {
                k: v.to(self.device) 
                for k, v in self.next_batch.items()
            }
    
    def __iter__(self):
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        """Get next batch (already on GPU)."""
        if self.stream is not None:
            # Wait for async transfer to complete
            torch.cuda.current_stream().wait_stream(self.stream)
        
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        
        # Start loading next batch while current one is being processed
        self.preload()
        
        return batch
    
    def __len__(self):
        """Return length of underlying loader if available."""
        return len(self.loader) if hasattr(self.loader, '__len__') else 0


def enable_prefetching(use_prefetcher: bool = True):
    """
    Decorator/helper to optionally enable prefetching.
    
    Usage in trainer:
        for batch in enable_prefetching_if_needed(dataloader, device):
            # batch is already on device
            ...
    """
    def wrapper(loader, device):
        if use_prefetcher and torch.cuda.is_available():
            return DataPrefetcher(loader, device)
        else:
            # Fallback: manual transfer
            class ManualTransfer:
                def __init__(self, loader, device):
                    self.loader = loader
                    self.device = device
                
                def __iter__(self):
                    for batch in self.loader:
                        yield {k: v.to(self.device, non_blocking=True) 
                               for k, v in batch.items()}
                
                def __len__(self):
                    return len(self.loader)
            
            return ManualTransfer(loader, device)
    
    return wrapper


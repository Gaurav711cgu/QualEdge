import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class VisionCalibrationDataset(Dataset):
    def __init__(self, num_samples: int = 1024, input_shape=(3, 224, 224), ood: bool = False):
        self.num_samples = num_samples
        self.input_shape = input_shape
        self.ood = ood
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.ood:
            # Out of distribution: random uniform noise, no natural image patterns
            x = torch.rand(self.input_shape)
        else:
            # Normal distribution: simulated normalized natural images (mean/std)
            # Standard ImageNet mean/std simulation
            x = torch.randn(self.input_shape) * 0.2 + 0.5
            x = torch.clamp(x, 0.0, 1.0)
        y = torch.tensor(idx % 1000, dtype=torch.long)
        return x, y

class AudioCalibrationDataset(Dataset):
    def __init__(self, num_samples: int = 256, max_len: int = 3000, ood: bool = False):
        self.num_samples = num_samples
        self.max_len = max_len
        self.ood = ood
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.ood:
            # Extreme high dynamic range / white noise calibration
            # Specially chosen to demonstrate audio dynamic range sensitivity breakdown
            features = torch.randn(80, 3000) * 10.0
        else:
            # Simulated Mel Spectrogram features
            features = torch.randn(80, 3000) * 1.5 - 2.0
        return features

class TextCalibrationDataset(Dataset):
    def __init__(self, num_samples: int = 512, seq_len: int = 512, ood: bool = False):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.ood = ood
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        if self.ood:
            # Out-of-distribution text calibration (random token IDs or repeating garbage)
            # Highly concentrated on a single token, which fails to represent LLM weight activations
            input_ids = torch.full((self.seq_len,), 42, dtype=torch.long)
        else:
            # Typical text tokens from WikiText-103 representation
            input_ids = torch.randint(0, 32000, (self.seq_len,), dtype=torch.long)
        
        # Self-attention mask
        attention_mask = torch.ones((self.seq_len,), dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def get_calibration_loader(model_family: str, num_samples: int, batch_size: int, ood: bool = False) -> DataLoader:
    if model_family == "vision":
        dataset = VisionCalibrationDataset(num_samples=num_samples, ood=ood)
    elif model_family == "audio":
        dataset = AudioCalibrationDataset(num_samples=num_samples, ood=ood)
    elif model_family == "language":
        dataset = TextCalibrationDataset(num_samples=num_samples, ood=ood)
    else:
        raise ValueError(f"Unknown model family: {model_family}")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

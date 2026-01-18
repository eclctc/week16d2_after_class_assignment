import torch
import torch.nn as nn

class LearnedAffine(nn.Module):
    """Custom layer that learns scale and shift parameters."""
    def __init__(self, num_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(num_features))
        self.shift = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return x * self.scale + self.shift

if __name__ == "__main__":
    # Sanity Test for LearnedAffine
    print("Running Sanity Test for LearnedAffine...")
    
    # 1. Check instantiation
    layer = LearnedAffine(num_features=10)
    print(" - Instantiated layer successfully.")
    
    # 2. Check forward pass
    input_tensor = torch.randn(5, 10) # Batch size 5, features 10
    output = layer(input_tensor)
    print(f" - Output shape checked: {output.shape}")
    assert output.shape == input_tensor.shape, "Output shape mismatch!"
    
    # 3. Check parameters
    print(f" - Parameters: scale={layer.scale.shape}, shift={layer.shift.shape}")
    assert layer.scale.requires_grad, "Scale should require grad"
    assert layer.shift.requires_grad, "Shift should require grad"

    print("Sanity Test Passed! ✅")
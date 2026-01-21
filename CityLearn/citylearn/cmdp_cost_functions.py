import torch


def placeholder_cost_fn(obs: torch.Tensor) -> torch.Tensor:
    # Placeholder cost function that returns zero cost
    return torch.tensor(0.0, dtype=torch.float32)
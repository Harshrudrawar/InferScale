"""Small fixed-shape tensor graph to validate the compiler/runtime installation."""


def create():
    import torch

    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 8))
    return model.eval(), (torch.randn(4, 16),)

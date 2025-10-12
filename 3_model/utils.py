import torch


def display_matrix_dimension(x: torch.Tensor, x_name: str = "") -> None:
    """Describe number of rows and columns of tensor x."""
    rows = x.shape[0]
    cols = x.shape[1]
    if x_name:
        print(f">>> {x_name} dimensions: {rows} rows, {cols} columns")
        return
    print(f">>> Dimensions: {rows} rows, {cols} columns")

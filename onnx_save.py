import torch


def export(model: torch.nn.Module,
           output_name: str,
           input_size: int,
           batch_size: int = -1):
    model = model.to("cpu")
    if batch_size == -1:
        dummy_input = torch.randn(1, input_size)
        dynamic_axes = {'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}}
    else:
        dummy_input = torch.randn(batch_size, input_size)
        dynamic_axes = {}
    torch.onnx.export(model, dummy_input, output_name, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)

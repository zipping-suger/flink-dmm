import torch


def export(model: torch.nn.Module,
           output_name: str,
           input_size: int):
    model = model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, input_size)
    dynamic_axes = {'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}}
    torch.onnx.export(model, dummy_input, output_name, input_names=['input'], output_names=['output'], dynamic_axes=dynamic_axes)

def export_lstm(model: torch.nn.Module,
                output_name: str,
                input_size: int,
                hidden_size: int,
                lstm_num_layer: int):
    model = model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, 1, input_size)
    dummy_h0 = torch.randn(lstm_num_layer, 1, hidden_size)
    dummy_c0 = torch.randn(lstm_num_layer, 1, hidden_size)

    res = model(dummy_input, (dummy_h0, dummy_c0))
    dynamic_axes = {'input': {1: 'sequence'},
                    'output': {1: 'sequence'}}
    torch.onnx.export(model, 
                      (dummy_input, (dummy_h0, dummy_c0)), 
                      output_name, 
                      input_names=['input', 'h0', 'c0'], 
                      output_names=['output', 'hn', 'cn'], 
                      dynamic_axes=dynamic_axes)

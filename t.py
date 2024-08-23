
# Using torch.compile with 'zentorch' as backend
import torch
import zentorch
compiled_model = torch.compile(model, backend='zentorch')
with torch.no_grad():
    output = compiled_model(input)

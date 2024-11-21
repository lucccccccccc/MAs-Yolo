import torch
# from torchstat import stat
from yolo import YOLO

model = YOLO()
model = YOLO.generate(model, onnx=False)
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))
from utils.benchmark import *
import torch 

model = torch.load('models/FireResNet50-97.pt')

dummy_input = torch.randn(1, 3, 224, 224).cuda()
benchmark(model, dummy_input)
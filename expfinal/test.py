import torch
from model_classes.m_2ch import MODEL_2CH
from model_classes.m_2chstream import MODEL_2CH2STREAM
model=MODEL_2CH()


from torchsummary import summary
from tensorboardX import SummaryWriter

writer = SummaryWriter()
writer.add_graph(model, (torch.rand([1, 2, 64, 64])))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
summary(model, input_size=(2, 64, 64))

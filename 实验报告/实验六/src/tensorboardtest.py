from tensorboardX import SummaryWriter

writer = SummaryWriter()
writer.add_scalar('exponential', 3, global_step=3)

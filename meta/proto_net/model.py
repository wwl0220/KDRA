import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def euclidean_dist(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class Protonet(nn.Module):
    def __init__(self, encoder):
        super(Protonet, self).__init__()

        self.encoder = encoder

    def loss(self, sample, is_cuda):
        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        xs = Variable(sample['xs'])
        xs = xs.to(device)
        xq = Variable(sample['xq'])
        xq = xq.to(device)

        n_class = xs.size(0)
        assert xq.size(0) == n_class
        n_support = xs.size(1)  # 支持集中每类样本的样本数
        n_query = xq.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if xq.is_cuda:
            target_inds = target_inds.cuda()

        x = torch.cat([xs.contiguous().view(n_class * n_support, *xs.size()[2:]),

                       xq.contiguous().view(n_class * n_query, *xq.size()[2:])])

        z = self.encoder.forward(x)
        z_dim = z.size(-1)

        z_proto = z[:n_class * n_support].view(n_class, n_support, z_dim).mean(1)
        zq = z[n_class * n_support:]
        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return z_proto, loss_val, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }

    def test_predict(self, eval_data, z_proto, is_cuda):
        if is_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        eval_data = eval_data.to(device)

        n_class = eval_data.size(0)
        n_size = eval_data.size(1)

        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_size, 1).long()
        target_inds = Variable(target_inds, requires_grad=False)

        if eval_data.is_cuda:
            target_inds = target_inds.cuda()

        target = target_inds.squeeze()

        x = eval_data.contiguous().view(n_class * n_size, *eval_data.size()[2:])

        z = self.encoder.forward(x)

        dists = euclidean_dist(z, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_size, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target).float().mean()

        return y_hat, target, {
            'loss': loss_val.item(),
            'acc': acc_val.item()
        }


def load_protonet(x_dim, hid_dim, z_dim):
    def dnn_block(in_sizes, out_sizes):
        return nn.Sequential(
            nn.Linear(in_sizes, out_sizes),
            nn.ReLU()
        )

    encoder = nn.Sequential(
        dnn_block(x_dim, hid_dim),
        dnn_block(hid_dim, hid_dim),
        dnn_block(hid_dim, z_dim)
    )

    return Protonet(encoder)


def evaluate(model, val_data, meters, is_cuda):
    model.eval()  # 将模型切换到评估模式

    for field, meter in meters.items():
        meter.reset()

    proto = 0
    for sample in val_data:
        proto, _, output = model.loss(sample, is_cuda)
        for field, meter in meters.items():
            meter.add(output[field])

    return proto

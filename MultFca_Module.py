from torch import nn
import torch, math
import torch.nn.functional as F


def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width, height, channel, fidx_u=[0,0,6,0], fidx_v=[0,1,0,5]):
    # width, height, channel: input size
    # fidx_u: horizontal indices of selected frequency
    scale_ratio = width // 10   # 7, 2, 4, 8
    fidx_u = [u * scale_ratio for u in fidx_u]
    fidx_v = [v * scale_ratio for v in fidx_v]

    dct_weights = torch.zeros(1, len(fidx_u), width, height)

    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i, t_x, t_y] = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)

    sum_weights0 = torch.sum(torch.sum(dct_weights, dim=2), dim=2)

    ids0 = torch.abs(sum_weights0) < 1e-4
    ids1 = torch.abs(sum_weights0) > 1e-4

    sum_weights0[ids0] = torch.mean(sum_weights0[ids1])
    sum_weights = sum_weights0.view(1, len(fidx_u), 1, 1).expand_as(dct_weights)
    dct_weights = dct_weights / sum_weights

    return dct_weights


class MultFcaLayer(nn.Module):
    def __init__(self, channel, reduction=16, width=208, height=208):
        super(MultFcaLayer, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(self.width, self.height, channel))
        self.fc = nn.Sequential(
            nn.Linear(channel*self.pre_computed_dct_weights.size()[1], channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (self.height, self.width))

        n = self.pre_computed_dct_weights.size()[1]
        ys = torch.zeros(b, c*n)
        for i in range(n):
            ys[:, i*c:(i+1)*c] = torch.sum(y*self.pre_computed_dct_weights[:,i:i+1,:,:], dim=(2, 3))

        y = self.fc(ys.cuda()).view(b, c, 1, 1)
        return x * y.expand_as(x)



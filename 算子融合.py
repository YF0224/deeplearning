import torch
import torch.nn.functional as F
import torch.nn as nn

in_channels = 2
out_channels = 2
kernel_size = 3
w = 9
h = 9

x = torch.ones(1, in_channels, w, h)

# 原始写法
conv_2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv_2d_pointwise = nn.Conv2d(in_channels, out_channels,1)
result1 = conv_2d(x) + conv_2d_pointwise(x) + x
# print(result1)
# print(x.shape)

# 算子融合
# 1)改造
pointwise_to_conv_weight = F.pad(conv_2d_pointwise.weight, [1, 1, 1, 1, 0, 0, 0, 0])# 2*2*1*1->2*2*3*3
conv_2d_for_pointwise = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv_2d_for_pointwise.weight = nn.Parameter(pointwise_to_conv_weight)
conv_2d_for_pointwise.bias = conv_2d_pointwise.bias

zeros = torch.unsqueeze(torch.zeros(kernel_size, kernel_size), 0)
stars = torch.unsqueeze(F.pad(torch.ones(1, 1), [1, 1, 1, 1]), 0)
satrs_zeros = torch.unsqueeze(torch.cat([stars, zeros], 0), 0)
zeros_stars = torch.unsqueeze(torch.cat([zeros, stars], 0), 0)
identity_to_conv_weight = torch.cat([satrs_zeros, zeros_stars], 0)
identity_to_conv_bias = torch.zeros(out_channels)
conv_2d_for_identity = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv_2d_for_identity.weight = nn.Parameter(identity_to_conv_weight)
conv_2d_for_identity.bias = nn.Parameter(identity_to_conv_bias)

result2 = conv_2d(x) + conv_2d_for_pointwise(x) + conv_2d_for_identity(x)
# print(torch.isclose(result1, result2))
# 2)融合
conv_2d_for_fusion = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
conv_2d_for_fusion.weight = nn.Parameter(conv_2d.weight.data + conv_2d_for_pointwise.weight.data + conv_2d_for_identity.weight.data)
conv_2d_for_fusion.bias = nn.Parameter(conv_2d.bias.data + conv_2d_for_pointwise.bias.data + conv_2d_for_identity.bias.data)

result3 = conv_2d_for_fusion(x)
print(torch.isclose(result3, result2))
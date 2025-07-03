import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from embedding import BertEmbedding
from torchsummary import summary


class Attention(nn.Module):
    def forward(self, query, key, value, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        p_att = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_att = dropout(p_att)
        return torch.matmul(p_att, value), p_att


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        self.d_k = d_model // h
        self.h = h
        self.fc = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.outputlayer = nn.LayerNorm(d_model, eps=1e-6)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        input = query
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.fc, (query, key, value))]
        x, att = self.attention(query, key, value, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        x = self.outputlayer(x + input)
        return x


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.outlayer = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        output = self.w_2(self.dropout(self.activation(self.w_1(x))))
        return self.outlayer(output + x)


class TransformerEncoder(nn.Module):
    def __init__(self, hidden, att_heads, feed_forward_hidden, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadedAttention(h=att_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        # self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        # self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.attention(x, x, x)
        x = self.feed_forward(x)
        return self.dropout(x)


class Classification(nn.Module):
    def __init__(self, hidden, classes):
        super(Classification, self).__init__()
        self.fc = nn.Linear(hidden, classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class Bert(nn.Module):
    def __init__(self, word_size=128, hidden=512, n_layers=6, att_heads=8, dropout=0):
        super(Bert, self).__init__()
        self.embedding = BertEmbedding(word_size, embed_size=hidden)
        self.transformer_encoders = nn.ModuleList([TransformerEncoder(hidden, att_heads, hidden * 4, dropout)
                                                   for _ in range(n_layers)])
        self.classification = Classification(hidden, 25)

    def forward(self, x):
        x = self.embedding(x)
        for transformer in self.transformer_encoders:
            x = transformer.forward(x)
        # y = self.embedding(y)
        # for transformer in self.transformer_encoders:
        #     y = transformer.forward(y)
        return x

class AlBert(nn.Module):
    def __init__(self, word_size=128, hidden=512, n_layers=6, att_heads=8, dropout=0):
        super(AlBert, self).__init__()
        self.embedding = BertEmbedding(word_size, embed_size=hidden)
        self.transformer_encoders = TransformerEncoder(hidden, att_heads, hidden * 4, dropout)
        self.n_layers = n_layers

    def forward(self, x):
        x = self.embedding(x)
        for i in range(self.n_layers):
            x = self.transformer_encoders.forward(x)
        return x

class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(res_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=1, stride=stride)
        else:
            self.conv3 = None

    def forward(self, x):
        output1 = F.relu(self.bn1(self.conv1(x)))
        output1 = self.bn2(self.conv2(output1))
        if self.conv3:
            output2 = self.conv3(x)
            output = F.relu(output1 + output2)
            return output
        else:
            output = F.relu(output1)
        return output


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, stride=2, padding=3),
                                   nn.BatchNorm1d(64),
                                   nn.ReLU(),
                                   nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        self.res_block1 = nn.Sequential(res_block(64, 64, stride=1),
                                        res_block(64, 64, stride=1))

        self.res_block2 = nn.Sequential(res_block(64, 128, use_1x1conv=True, stride=2),
                                        res_block(128, 128, stride=1))

        self.res_block3 = nn.Sequential(res_block(128, 256, use_1x1conv=True, stride=2),
                                        res_block(256, 256, stride=1))

        self.res_block4 = nn.Sequential(res_block(256, 512, use_1x1conv=True, stride=2),
                                        res_block(512, 512, stride=1))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, 25)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        in_size = x.size(0)
        output = self.conv1(x)
        output = self.res_block1(output)
        output = self.res_block2(output)
        output = self.res_block3(output)
        output = self.res_block4(output)
        output = self.avgpool(output)
        output = output.view(in_size, -1)
        # output = self.fc(output)
        # output = self.softmax(output)
        return output


def distillation_loss(outputs_teacher, outputs_student, labels, temperature, alpha):
    soft_teacher = F.softmax(outputs_teacher / temperature, dim=1)
    soft_student = F.log_softmax(outputs_student / temperature, dim=1)
    loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature ** 2)
    classification_loss = F.cross_entropy(outputs_student, labels)
    total_loss = alpha * loss + (1 - alpha) * classification_loss
    return total_loss


def contrastiveloss(x, y, temperature=0.1):
    batchsize = x.size(0)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    similarity_matrix = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=2)
    mask = torch.eye(batchsize, device=similarity_matrix.device)
    positive_mask = mask.bool()
    positive = similarity_matrix[positive_mask].unsqueeze(-1)
    negative_mask = ~positive_mask.bool()
    negative = similarity_matrix[negative_mask].reshape(batchsize, -1)
    logits = torch.cat([positive, negative], dim=1)
    logits = (logits + 1) / 2
    logits /= temperature
    labels = torch.zeros(batchsize, device=similarity_matrix.device).long()
    return F.cross_entropy(logits, labels)


# MobileNetv3
# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     :param v:
#     :param divisor:
#     :param min_value:
#     :return:
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
#
# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(SELayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, _make_divisible(channel // reduction, 8)),
#             nn.ReLU(inplace=True),
#             nn.Linear(_make_divisible(channel // reduction, 8), channel),
#             h_sigmoid()
#         )
#
#     def forward(self, x):
#         #b, c, _, _ = x.size()
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#         return x * y
#
#
# def conv_3x3_bn(inp, oup, stride):
#     return nn.Sequential(
#         nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
#         nn.BatchNorm1d(oup),
#         h_swish()
#     )
#
#
# def conv_1x1_bn(inp, oup):
#     return nn.Sequential(
#         nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
#         nn.BatchNorm1d(oup),
#         h_swish()
#     )
#
#
# class InvertedResidual(nn.Module):
#     def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
#         super(InvertedResidual, self).__init__()
#         assert stride in [1, 2]
#
#         self.identity = stride == 1 and inp == oup
#
#         if inp == hidden_dim:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
#                           bias=False),
#                 nn.BatchNorm1d(hidden_dim),
#                 h_swish() if use_hs else nn.ReLU(inplace=True),
#                 # Squeeze-and-Excite
#                 SELayer(hidden_dim) if use_se else nn.Identity(),
#                 # pw-linear
#                 nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm1d(oup),
#             )
#         else:
#             self.conv = nn.Sequential(
#                 # pw
#                 nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm1d(hidden_dim),
#                 h_swish() if use_hs else nn.ReLU(inplace=True),
#                 # dw
#                 nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
#                           bias=False),
#                 nn.BatchNorm1d(hidden_dim),
#                 # Squeeze-and-Excite
#                 SELayer(hidden_dim) if use_se else nn.Identity(),
#                 h_swish() if use_hs else nn.ReLU(inplace=True),
#                 # pw-linear
#                 nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
#                 nn.BatchNorm1d(oup),
#             )
#
#     def forward(self, x):
#         if self.identity:
#             return x + self.conv(x)
#         else:
#             return self.conv(x)
#
#
# class MobileNetV3(nn.Module):
#     def __init__(self, cfg, mode, num_classes=25, width_mult=1.):
#         super(MobileNetV3, self).__init__()
#         # setting of inverted residual blocks
#         self.cfg = cfg
#         assert mode in ['large', 'small']
#
#         # building first layer
#         input_channel = _make_divisible(16 * width_mult, 8)
#         layers = [conv_3x3_bn(2, input_channel, 2)]
#         # building inverted residual blocks
#         block = InvertedResidual
#         for k, t, c, use_se, use_hs, s in self.cfg:
#             output_channel = _make_divisible(c * width_mult, 8)
#             exp_size = _make_divisible(input_channel * t, 8)
#             layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
#             input_channel = output_channel
#         self.features = nn.Sequential(*layers)
#         # building last several layers
#         self.conv = conv_1x1_bn(input_channel, exp_size)
#         self.avgpool = nn.AdaptiveAvgPool1d(1)
#         output_channel = {'large': 1280, 'small': 1024}
#         output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
#             mode]
#         self.classifier = nn.Sequential(
#             nn.Linear(exp_size, output_channel),
#             h_swish(),
#             nn.Dropout(0.2),
#             nn.Linear(output_channel, num_classes),
#         )
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.conv(x)
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         # x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()

#EfficientNetv2
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, _make_divisible(inp // reduction, 8)),
            SiLU(),
            nn.Linear(_make_divisible(inp // reduction, 8), oup),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm1d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv1d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm1d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv1d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv1d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv1d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv1d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm1d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(2, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = 512
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()

# summary(ResNet().to('cuda'), (2, 4096), batch_size=3)

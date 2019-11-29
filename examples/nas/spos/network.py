import torch
import torch.nn as nn

from blocks import ShuffleNetBlock, ShuffleXceptionBlock

from nni.nas.pytorch import mutables


class ShuffleNetV2OneShot(nn.Module):

    def __init__(self, input_size=224, first_conv_channels=16, last_conv_channels=1024, n_classes=1000):
        super().__init__()

        assert input_size % 32 == 0

        self.stage_blocks = [4, 4, 8, 4]
        self.stage_channels = [64, 160, 320, 640]

        # building first layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, first_conv_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(first_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )

        p_channels = first_conv_channels
        features = []
        for num_blocks, channels in zip(self.stage_blocks, self.stage_channels):
            features.extend(self._make_blocks(num_blocks, p_channels, channels))
            p_channels = channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(p_channels, last_conv_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(last_conv_channels, affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(last_conv_channels, n_classes, bias=False)

        self._initialize_weights()

    def _make_blocks(self, blocks, in_channels, channels):
        result = []
        for i in range(blocks):
            stride = 2 if i == 0 else 1
            inp = in_channels if i == 0 else channels // 2
            oup = channels

            base_mid_channels = channels // 2
            mid_channels = int(base_mid_channels)  # prepare for scale
            result.append(mutables.LayerChoice([
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=3, stride=stride),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=5, stride=stride),
                ShuffleNetBlock(inp, oup, mid_channels=mid_channels, ksize=7, stride=stride),
                ShuffleXceptionBlock(inp, oup, mid_channels=mid_channels, stride=stride)
            ]))
        return result

    def forward(self, x):
        bs = x.size(0)
        x = self.first_conv(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(bs, -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    # architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    # scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
    model = ShuffleNetV2_OneShot()
    # print(model)

    test_data = torch.rand(5, 3, 224, 224)
    test_outputs = model(test_data)
    print(test_outputs.size())

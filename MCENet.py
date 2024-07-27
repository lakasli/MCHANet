import torch
import torch.nn as nn
from torchinfo import summary


class LinearTransform(nn.Module):
    def __init__(self, in_channels):
        super(LinearTransform, self).__init__()
        self.linear_q = nn.Linear(in_channels, 64)
        self.linear_k = nn.Linear(in_channels, 64)
        self.linear_v = nn.Linear(in_channels, 64)

    def forward(self, x):
        q = self.linear_q(x.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.linear_k(x.permute(0, 2, 1)).permute(0, 2, 1)
        v = self.linear_v(x.permute(0, 2, 1)).permute(0, 2, 1)
        return q, k, v


class MultiScaleConv(nn.Module):
    def __init__(self):
        super(MultiScaleConv, self).__init__()
        self.dwconv3 = nn.Conv1d(64, 64, kernel_size=3, groups=64, padding=1)
        self.gconv1_3 = nn.Conv1d(64, 64, kernel_size=1)
        self.dwconv5 = nn.Conv1d(64, 64, kernel_size=5, groups=64, padding=2)
        self.gconv1_5 = nn.Conv1d(64, 64, kernel_size=1)
        self.relu_linear_attention_3 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        self.relu_linear_attention_5 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, k, v):
        conv3 = self.dwconv3(k)
        conv3 = self.gconv1_3(conv3)
        k = self.relu_linear_attention_3(conv3.permute(0, 2, 1)).permute(0, 2, 1)

        conv5 = self.dwconv5(v)
        conv5 = self.gconv1_5(conv5)
        v = self.relu_linear_attention_5(conv5.permute(0, 2, 1)).permute(0, 2, 1)

        return k, v


class MSLA(nn.Module):
    def __init__(self, in_channels, out_features):
        super(MSLA, self).__init__()
        self.linear_transform = LinearTransform(in_channels)
        self.multi_scale_conv = MultiScaleConv()
        self.output_linear = nn.Linear(192, 128)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        q, k, v = self.linear_transform(x)
        k, v = self.multi_scale_conv(k, v)
        combined_att = torch.cat([q, k, v], dim=1)
        x = self.output_linear(combined_att.permute(0, 2, 1))
        x = self.global_max_pool(x.permute(0, 2, 1))
        x = x.view(x.size(0), -1)
        return x


class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvFeatureExtractor, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, (2, 7), padding=(0, 3))
        self.conv1_2 = nn.Conv1d(1, 64, 7, padding=3)
        self.conv1_3 = nn.Conv1d(1, 64, 7, padding=3)
        self.conv1_4 = nn.Conv1d(1, 128, 7, padding=3)
        self.conv1_5 = nn.Conv1d(1, 128, 7, padding=3)
        self.conv1_6 = nn.Conv1d(1, 128, 7, padding=3)
        self.relu = nn.ReLU()

    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = self.conv1_1(x1)
        x1 = self.relu(x1)

        x2 = self.conv1_2(x2)
        x2 = self.relu(x2)

        x3 = self.conv1_3(x3)
        x3 = self.relu(x3)

        x4 = self.conv1_4(x4)
        x4 = self.relu(x4)

        x5 = self.conv1_5(x5)
        x5 = self.relu(x5)

        x6 = self.conv1_6(x6)
        return x1, x2, x3, x4, x5, x6


class LSTMFeatureExtractor(nn.Module):
    def __init__(self):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm1 = nn.LSTM(128, 128, num_layers=2, batch_first=True)
        self.attention = nn.Linear(128, 128)

    def forward(self, y1):
        y1, _ = self.lstm1(y1)
        w = torch.softmax(self.attention(y1), dim=-1)
        y1 = torch.sum(y1 * w, dim=1)
        return y1


class MCENet(nn.Module):
    def __init__(self, num_classes):
        super(MCENet, self).__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        self.conv_feature_extractor = ConvFeatureExtractor()
        self.msla = MSLA(128, 512)
        self.conv2 = nn.Conv2d(64, 90, (1, 7), padding=(0, 3))
        self.conv3 = nn.Conv2d(90, 128, (3, 5), padding=(0, 2))
        self.conv6 = nn.Conv2d(128, 128, (1, 7), padding=(0, 3))
        self.conv7 = nn.Conv2d(128, 128, (2, 5), padding=(0, 2))
        self.lstm_feature_extractor = LSTMFeatureExtractor()
        self.fc1 = nn.Linear(384, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x1 = torch.unsqueeze(x, dim=1)
        x2 = x[:, 0, :].unsqueeze(1)
        x3 = x[:, 1, :].unsqueeze(1)
        x4 = torch.atan2(x2, x3)
        x5 = torch.sqrt(x2 ** 2 + x3 ** 2)
        x6 = torch.abs(torch.fft.fft(x5))
        x6 = x6[:, :, :x6.shape[-1] // 2]

        x1, x2, x3, x4, x5, x6 = self.conv_feature_extractor(x1, x2, x3, x4, x5, x6)

        y1 = torch.cat((x4.unsqueeze(2), x5.unsqueeze(2)), dim=2)
        y1 = self.conv6(y1)
        y1 = self.relu(y1)
        y1 = self.conv7(y1)
        y1 = self.relu(y1)
        y1 = torch.squeeze(y1, dim=2)
        y1 = y1.permute(0, 2, 1)
        y1 = self.lstm_feature_extractor(y1)

        y2 = torch.cat((x1, x2.unsqueeze(2), x3.unsqueeze(2)), dim=2)
        y2 = self.conv2(y2)
        y2 = self.relu(y2)
        y2 = self.conv3(y2)
        y2 = self.relu(y2)
        y2 = torch.squeeze(y2, dim=2)
        y2 = y2.permute(0, 2, 1)
        y2 = self.lstm_feature_extractor(y2)

        y3 = self.msla(x6)
        y3 = self.relu(y3)

        x = torch.cat((y1, y2, y3), dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MCENet(num_classes=24).to(device)
    summary(model, (64, 2, 1024))

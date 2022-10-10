import torch
import torch.nn as nn
def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias)

class Autoencoder(nn.Module):
    def __init__(self, previous_num):
        super(Autoencoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=previous_num, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.decv1 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.decv2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decv3 = nn.ConvTranspose2d(in_channels=64 + 64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.decv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decv5 = nn.ConvTranspose2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.decv6 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.apply(weight_init)

    def forward(self, x):

        # x: B,H,W,previous_seq --> B,previous_seq,H,W
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        conv2 = x.clone()
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        conv4 = x.clone()
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.decv1(x))
        x = self.relu(self.decv2(x))
        x = torch.cat([x, conv4], 1)
        x = self.relu(self.decv3(x))
        x = self.relu(self.decv4(x))
        x = torch.cat([x, conv2], 1)
        x = self.relu(self.decv5(x))
        x = self.tanh(self.decv6(x))
        # x: B,1,H,W --> B,H,W,1
        out = x.permute(0, 2, 3, 1).contiguous()
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        )
        self.apply(weight_init)
    def forward(self, x):
        out = self.encoder(x)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Tanh()
        )
        self.apply(weight_init)
    def forward(self, x):
        out = self.decoder(x)
        return out

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, kernel_size, filters, padding):
        super(ConvLSTM, self).__init__()
        self.filters = filters
        self.inputs_after_conv = nn.Conv2d(in_channels=in_channels, out_channels=4 * filters, kernel_size=kernel_size, stride=1, padding=padding)
        self.hidden_after_conv = nn.Conv2d(in_channels=filters, out_channels=4 * filters, kernel_size=kernel_size, stride=1, padding=padding)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.apply(weight_init)

    def forward(self, inputs, h, c):
        xi, xc, xf, xo = self.inputs_after_conv(inputs).split(self.filters, dim=1)
        hi, hc, hf, ho = self.hidden_after_conv(h).split(self.filters, dim=1)
        it = self.sigmoid(xi + hi)
        ft = self.sigmoid(xf + hf)
        new_c = (ft * c) + (it * self.tanh(xc + hc))
        ot = self.sigmoid(xo + ho)
        new_h = ot * self.tanh(new_c)
        return new_h, new_c
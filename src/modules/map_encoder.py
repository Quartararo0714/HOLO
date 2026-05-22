from .module_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 32, bilinear))

        self.outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return x, logits
    

class MapEncoder(nn.Module):
    def __init__(self, embedding_dim=16, out_channels=2, bilinear=True):
        super(MapEncoder, self).__init__()
        self.embedding_roads = nn.Embedding(2, embedding_dim)
        self.embedding_buildings = nn.Embedding(2, embedding_dim)
        in_channels = 2 * embedding_dim
        self.encoder = UNet(in_channels, out_channels, bilinear)

    def forward(self, x):
        x_roads = self.embedding_roads(x[:, 0])
        x_buildings = self.embedding_buildings(x[:, 1])
        x = torch.cat([x_roads, x_buildings], dim=-1).permute(0, 3, 1, 2)
        x, logits = self.encoder(x)
        return x, logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MapEncoder(embedding_dim=16, out_channels=2, bilinear=True).to(device)
    input_tensor = torch.randint(0, 2, (4, 2, 256, 256)).to(device)  # Example input
    output, logits = model(input_tensor)
    print(output.shape)  # Should be (4, 64, 256, 256)
    print(logits.shape)  # Should be (4, 2, 256, 256)
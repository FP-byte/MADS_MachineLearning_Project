import math

import torch
from loguru import logger
from torch import Tensor, nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = config["hidden"]
        self.convolutions = nn.ModuleList(
            [
                ConvBlock(1, hidden),
            ]
        )

        for i in range(config["num_blocks"]):
            self.convolutions.extend([ConvBlock(hidden, hidden)])
        self.convolutions.append(nn.MaxPool2d(2, 2))

        activation_map_size = config["shape"][0] // 2 * config["shape"][1] // 2
        logger.info(f"Activation map size: {activation_map_size}")
        logger.info(f"Input linear: {activation_map_size * hidden}")

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(activation_map_size * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, config["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)
        x = self.dense(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_seq_len, d_model)
        # batch, seq_len, d_model
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        # feel free to change the input parameters of the constructor
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        identity = x.clone()  # skip connection
        x, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + identity)  # Add & Norm skip
        identity = x.clone()  # second skip connection
        x = self.ff(x)
        x = self.layer_norm2(x + identity)  # Add & Norm skip
        return x

# transformer 1D
class Transformer(nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config["hidden"], config["num_heads"], config["dropout"]
                )
                for _ in range(config["num_blocks"])
            ]
        )

        self.out = nn.Linear(config["hidden"], config["num_classes"])

    def forward(self, x: Tensor) -> Tensor:
        # streamer:         (batch, seq_len, channels)
        # conv1d:           (batch, channels, seq_len)
        # pos_encoding:     (batch, seq_len, channels)
        # attention:        (batch, seq_len, channels)
        x = self.conv1d(x.transpose(1, 2))  # flip channels and seq_len for conv1d
        x = self.pos_encoder(x.transpose(1, 2))  # flip back to seq_len and channels

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = x.mean(dim=1)  # Global Average Pooling
        x = self.out(x)
        return x

class GRUmodel(nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        print(config)
        self.rnn = nn.GRU(
            input_size=config["input_size"],
            hidden_size=int(config["hidden"]),
            dropout=config["dropout"],
            batch_first=True,
            num_layers=int(config["num_layers"]),
        )
        self.linear = nn.Linear(int(config["hidden"]), config["num_classes"])

    def forward(self, x):
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        # feel free to change the input parameters of the constructor
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        identity = x.clone() # skip connection
        x, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + identity) # Add & Norm skip
        identity = x.clone() # second skip connection
        x = self.ff(x)
        x = self.layer_norm2(x + identity) # Add & Norm skip
        return x

#werkend
class Transformer2D(nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        hidden = config["hidden"]
        self.convolutions = nn.ModuleList([
            ConvBlock(1, hidden),
        ])

        for i in range(config['num_blocks']):
            self.convolutions.extend([ConvBlock(hidden, hidden), nn.ReLU()])
        self.convolutions.append(nn.MaxPool2d(2, 2))

        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config["hidden"], config["num_heads"], config["dropout"])
            for _ in range(config["num_blocks"])
        ])

        #self.out = nn.Linear(config["hidden"], config["num_classes"])
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config["hidden"], config["hidden"]//2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden"]//2, config['num_classes']),
        )

    def forward(self, x: Tensor) -> Tensor:
        # streamer:         (batch, seq_len, channels)
        # conv2d            (batch, channels, seq_len) [32, 1, 16, 12]
        # conv1d:           (batch, channels, seq_len) [32, 1, 192]
        # pos_encoding:     (batch, seq_len, channels)
        # attention:        (batch, seq_len, channels)
        #x = self.conv1d(x.transpose(1, 2)) # flip channels and seq_len for conv1d
        x = x.view(32, 1, 16, 12) # reshape to 2D
        for conv in self.convolutions:
            x = conv(x)
        x = x.view(32, config["hidden"], (6*8))
        x = self.pos_encoder(x.transpose(1, 2)) # flip back to seq_len and channels

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = x.mean(dim=1) # Global Average Pooling
        x = self.out(x)
        return x

# ResNet Block for 1D convolutions
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection (identity map)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Add the shortcut (skip connection)
        out = self.relu(out)
        return out




class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Identity shortcut connection, could be a 1x1 convolution to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)  # Add the shortcut (skip connection)
        out = self.relu(out)
        return out

# Transformer 1D model with ResNet block
class Transformer1DResnet(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

        # Adding a ResNet Block after the convolution layer
        self.resnet_block = ResNetBlock(config["hidden"], config["hidden"])

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config["hidden"], config["num_heads"], config["dropout"]
                )
                for _ in range(config["num_blocks"])
            ]
        )

        self.out = nn.Linear(config["hidden"], config["num_classes"])

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, channels)
        # Conv1D transforms to (batch, channels, seq_len)
        x = self.conv1d(x.transpose(1, 2))  # flip channels and seq_len for conv1d

        # Apply ResNet block after convolution
        x = self.resnet_block(x)

        # Apply positional encoding
        x = self.pos_encoder(x.transpose(1, 2))  # flip back to seq_len and channels

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = x.mean(dim=1)  # Global Average Pooling
        x = self.out(x)
        return x

# Define the 2D ResNet Block
class ResNetBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If input and output sizes do not match, apply a projection (1x1 conv)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        # Identity shortcut connection, could be a 1x1 convolution to match dimensions
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # Store the input for the skip connection

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Add the input to the output (skip connection)
        x += self.projection(identity)
        x = self.relu(x)

        return x

# 2D CNN WITH RESNET BLOCKS
class CNN2DResNet(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        hidden = config['hidden']
        
        # Start with an initial convolutional block
        self.convolutions = nn.ModuleList([
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden)
        ])
        
        # Adding multiple ResNet blocks
        for i in range(config['num_blocks']):
            self.convolutions.append(ResNetBlock(hidden, hidden))
        
        # MaxPool at the end of convolutions
        self.convolutions.append(nn.MaxPool2d(2, 2))

        # Fully connected (dense) layers
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear((8*6) * hidden, hidden),  # Adjust the size according to your image dimensions
            nn.ReLU(),
            nn.Linear(hidden, config['num_classes']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.convolutions:
            x = layer(x)
        x = self.dense(x)
        return x


# 2D TRANSFORMER WITH RESNET BLOCKS
class Transformer2DResNet(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        # Modify the initial Conv2D to accept 1 input channels 
        self.conv2d = nn.Conv2d(
            in_channels=1,  # Adjusted input channels if needed
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        print('2d conv')

        # Add ResNet Block (2D)
        self.resnet_block = ResNetBlock2D(config["hidden"], config["hidden"])
        print('resnet block')

        # Positional Encoding for Transformer input
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])
        print('positional encoding')

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config["hidden"], config["num_heads"], config["dropout"])
            for _ in range(config["num_blocks"])
        ])

        # Final output layers
        self.out = nn.Sequential(
            nn.Linear(config["hidden"], config["hidden"] // 2),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden"] // 2, config["num_classes"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # print("input", x.shape)
        #x = x.view(32, 1, 16, 12) # reshape to 2D
        # Apply Conv2D to the input (convert from (batch, channels, height, width) to (batch, hidden, height, width))
        x = self.conv2d(x.transpose(1, 2))  # (batch, hidden, height//2, width//2)

        # Apply ResNet Block (2D)
        x = self.resnet_block(x)  # (batch, hidden, height//2, width//2)

        # Apply positional encoding (convert back to (batch, seq_len, channels))
        x = self.pos_encoder(x.flatten(2).transpose(1, 2))  # Flatten and transpose to (batch, seq_len, channels)

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Global Average Pooling
        x = x.mean(dim=1)  # (batch, hidden)

        # Final classification layers
        x = self.out(x)  # (batch, num_classes)
        return x

import math
from typing import Dict
import torch
from loguru import logger
from torch import Tensor, nn
import torch.nn.functional as F


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

class SqueezeExcite1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcite1D, self).__init__()
        # Squeeze: Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excite: Two fully connected layers (bottleneck architecture)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        
        # Sigmoid activation to get attention weights
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        x_se = self.global_avg_pool(x)  # (batch, channels, 1)
        x_se = x_se.view(x_se.size(0), -1)  # Flatten to (batch, channels)
        
        # Excite
        x_se = self.fc1(x_se)  # (batch, channels // reduction)
        x_se = self.relu(x_se)
        x_se = self.fc2(x_se)  # (batch, channels)
        
        # Scale input feature map with the learned attention weights
        x_se = self.sigmoid(x_se).unsqueeze(-1)  # (batch, channels, 1)
        
        return x * x_se  # Apply attention (channel recalibration)

class Transformer1DSE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        # Convolutional Layer
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        # Squeeze-and-Excitation Block
        self.se_block = SqueezeExcite1D(config["hidden"])

        # Positional Encoding for Transformer input
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(config["hidden"], config["num_heads"], config["dropout"])
                for _ in range(config["num_blocks"])
            ]
        )

        # Final output layer
        self.out = nn.Linear(config["hidden"], config["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Conv1D to the input (shape: batch, channels, seq_len)
        x = self.conv1d(x)  # (batch, hidden, seq_len)
        
        # Apply Squeeze-and-Excitation block (Channel recalibration)
        x = self.se_block(x)
        
        # Apply positional encoding
        x = self.pos_encoder(x.transpose(1, 2))  # (batch, seq_len, channels)
        
        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Final output layer
        x = x.mean(dim=1)  # Global Average Pooling (batch, hidden)
        x = self.out(x)  # (batch, num_classes)
        
        return x

class GRU(nn.Module):
    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        print(config)
        self.rnn = nn.GRU(
            input_size=config["input"],
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

class AttentionGRU(nn.Module):
    def __init__(
        self,
        config: Dict,
    ) -> None:
        super().__init__()
        self.rnn = nn.GRU(
            input_size=config["input"],
            hidden_size=config["hidden"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config["hidden"],
            num_heads=4,
            dropout=config["dropout"],
            batch_first=True,
        )
        self.linear = nn.Linear(config["hidden"], config["num_classes"])

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        x, _ = self.attention(x.clone(), x.clone(), x)
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
        self.config = config
        self.hidden = self.config["hidden"]
        self.convolutions = nn.ModuleList([
            ConvBlock(1, self.hidden),
        ])

        for i in range(config['num_blocks']):
            self.convolutions.extend([ConvBlock(self.hidden, self.hidden), nn.ReLU()])
        self.convolutions.append(nn.MaxPool2d(2, 2))

        self.pos_encoder = PositionalEncoding(self.hidden, self.config["dropout"])

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.config["hidden"], self.config["num_heads"], self.config["dropout"])
            for _ in range(self.config["num_blocks"])
        ])

        #self.out = nn.Linear(config["hidden"], config["num_classes"])
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.config["hidden"], self.config["hidden"]//2),
            nn.ReLU(),
            nn.Dropout(self.config["dropout"]),
            nn.Linear(self.config["hidden"]//2, self.config['num_classes']),
        )

    def forward(self, x: Tensor) -> Tensor:
        # streamer:         (batch, seq_len, channels)
        # conv2d            (batch, channels, seq_len) [32, 1, 16, 12]
        # conv1d:           (batch, channels, seq_len) [32, 1, 192]
        # pos_encoding:     (batch, seq_len, channels)
        # attention:        (batch, seq_len, channels)
        #x = self.conv1d(x.transpose(1, 2)) # flip channels and seq_len for conv1d
        x = x.view(x.size(0), 1, 16, 12) # reshape to 2D
        for conv in self.convolutions:
            x = conv(x)
        x = x.view(x.size(0), self.config["hidden"], (6*8))
        x = self.pos_encoder(x.transpose(1, 2)) # flip back to seq_len and channels

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        x = x.mean(dim=1) # Global Average Pooling
        x = self.out(x)
        return x


# ResNet Block for 1D convolutions
class ResNetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock1D, self).__init__()
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

# Transformer 1D model with ResNet block
# Transformer model with ResNet block
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
        self.resnet_block = ResNetBlock1D(config["hidden"], config["hidden"])

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

# Transformer with ResNet block and SE block 
class Transformer1DResnetSE(nn.Module):
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
        self.resnet_block = ResNetBlock1D(config["hidden"], config["hidden"])

        # Adding the SE block after ResNet Block
        self.se_block = SqueezeExcite1D(config["hidden"])

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

        # Apply Squeeze-and-Excitation block (channel recalibration)
        x = self.se_block(x)

        # Apply positional encoding
        x = self.pos_encoder(x.transpose(1, 2))  # flip back to seq_len and channels

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Global Average Pooling
        x = x.mean(dim=1)  # (batch, hidden)
        x = self.out(x)  # (batch, num_classes)
        
        return x

class Transformer1DResnetSEwithAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        # Convolutional Layer (1D Conv)
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        
        # ResNet Block
        self.resnet_block = ResNetBlock1D(config["hidden"], config["hidden"])

        # Squeeze-and-Excitation Block
        self.se_block = SqueezeExcite1D(config["hidden"])

        # Multi-Head Attention Layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=config["hidden"],
            num_heads=config["num_heads"],
            dropout=config["dropout"]
        )

        # Positional Encoding for Transformer input
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

        # Create multiple transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(config["hidden"], config["num_heads"], config["dropout"])
                for _ in range(config["num_blocks"])
            ]
        )

        # Final output layer
        self.out = nn.Linear(config["hidden"], config["num_classes"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply Conv1D to the input (shape: batch, channels, seq_len)
        x = self.conv1d(x)  # (batch, hidden, seq_len)
        
        # Apply ResNet Block (skip connection + convolution)
        x = self.resnet_block(x)
        
        # Apply Squeeze-and-Excitation block (Channel recalibration)
        x = self.se_block(x)
        
        # Reshape to match the input requirements for MultiHeadAttention (seq_len, batch, hidden)
        x = x.transpose(1, 2)  # (batch, hidden, seq_len) -> (seq_len, batch, hidden)
        
        # Apply Multi-Head Attention
        # self.multihead_attention expects input shape (seq_len, batch, embed_dim)
        attn_output, _ = self.multihead_attention(x, x, x)  # (seq_len, batch, hidden)
        
        # Add the attention output to the original input (Residual connection)
        x = x + attn_output  # Skip connection for attention
        
        # Apply positional encoding
        x = self.pos_encoder(x)  # (seq_len, batch, hidden)
        
        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Final output layer
        x = x.mean(dim=0)  # Global Average Pooling (seq_len, batch, hidden) -> (batch, hidden)
        x = self.out(x)  # (batch, num_classes)
        
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
        #print('2d conv')

        # Add ResNet Block (2D)
        self.resnet_block = ResNetBlock2D(config["hidden"], config["hidden"])
        print('resnet block')

        # Positional Encoding for Transformer input
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])
        #print('positional encoding')

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

    # - streamer:         (batch, seq_len, channels)
    # - conv1d:           (batch, channels, seq_len)
    # - pos_encoding:     (batch, seq_len, channels)
    # - gru (batchfirst): (batch, seq_len, channels)
    # - attention:        (batch, seq_len, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       # print("input", x.shape)
        x = x.view(x.size(0), 1, 16, 12) # reshape to 2D
       # print("input", x.shape)
        # Apply Conv2D to the input (convert from (batch, channels, height, width) to (batch, hidden, height, width))
        x = self.conv2d(x)  # (batch, hidden, height, width)
       # print("input afte cov2", x.shape)
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

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Global Average Pooling
        b, c, _, _ = x.size()  # batch_size, channels, height, width
        squeeze = F.adaptive_avg_pool2d(x, (1, 1))  # (b, c, 1, 1)
        squeeze = squeeze.view(b, c)  # Flatten to (b, c)

        # Excitation: Fully connected layers + sigmoid activation
        excitation = F.relu(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation)).view(b, c, 1, 1)  # (b, c, 1, 1)

        # Recalibration: Re-weight the original feature maps
        return x * excitation.expand_as(x)


class Transformer2DResNetSE(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()

        # Initial Conv2D Layer
        self.conv2d = nn.Conv2d(
            in_channels=1,  # Adjusted input channels if needed
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # ResNet Block (2D)
        self.resnet_block = ResNetBlock2D(config["hidden"], config["hidden"])

        # Squeeze-and-Excitation (SE) block after ResNet block
        self.se_block = SEBlock(config["hidden"])

        # Positional Encoding for Transformer input
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

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
        # Reshape input to 2D (batch, channels, height, width)
        x = x.view(x.size(0), 1, 16, 12)

        # Apply Conv2D to the input
        x = self.conv2d(x)  # (batch, hidden, height, width)

        # Apply ResNet Block (2D)
        x = self.resnet_block(x)  # (batch, hidden, height//2, width//2)

        # Apply SE Block to recalibrate features
        x = self.se_block(x)

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

class MultiHeadAttentionWithSE(nn.Module):
    def __init__(self, hidden_dim, num_heads, reduction=16, dropout=0.1):
        super(MultiHeadAttentionWithSE, self).__init__()
        
        # Multi-Head Attention layer
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        
        # Squeeze-and-Excitation block
        self.se_block = SqueezeExcitation(hidden_dim, reduction)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        
        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention mechanism with residual connection and layer normalization
        attn_output, _ = self.attn(x, x, x)  # (batch, seq_len, hidden)
        x = self.layer_norm1(x + self.dropout(attn_output))  # (batch, seq_len, hidden)
        
        # Apply Squeeze-and-Excitation block to recalibrate the channel attention
        x = self.se_block(x)  # Apply SE block on the output of attention

        # Feed-forward network with residual connection and layer normalization
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))

class Transformer2DResNetWithAttention(nn.Module):
    def __init__(self, config: dict) -> None:
        super(Transformer2DResNetWithAttention, self).__init__()

        self.conv2d = nn.Conv2d(in_channels=1, out_channels=config["hidden"], kernel_size=3, stride=2, padding=1)

        # Adding ResNet Block (already defined in your code)
        self.resnet_block = ResNetBlock2D(config["hidden"], config["hidden"])

        # Add the SE block after the ResNet Block
        self.se_block = SEBlock(config["hidden"])

        # Transformer-related components
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])
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
        x = x.view(x.size(0), 1, 16, 12)  # Reshape to 2D (batch, channels, height, width)

        # Apply Conv2D
        x = self.conv2d(x)

        # Apply ResNet Block
        x = self.resnet_block(x)

        # Apply Squeeze-and-Excite Block
        x = self.se_block(x)

        # Apply positional encoding and flatten the input for transformer
        x = self.pos_encoder(x.flatten(2).transpose(1, 2))

        # Pass through transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Final output layer
        x = self.out(x)
        return x
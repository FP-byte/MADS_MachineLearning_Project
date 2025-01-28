## Top Architectures Overview

### 1. 2D Convolutional Neural Network (CNN) with ResNet

**Architecture**:
| Layer Type         | Details                                                                                          |
|--------------------|--------------------------------------------------------------------------------------------------|
| **Convolutions**   | **ModuleList**                                                                                   |
| Conv2d             | 1 input channel, 128 output channels, kernel size 3x3, stride 1x1, padding 1x1                   |
| ReLU               | Activation function                                                                              |
| BatchNorm2d        | 128 channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True                     |
| **ResNetBlock2D**  |                                                                                                  |
| Conv2d             | 128 input/output channels, kernel size 3x3, stride 1x1, padding 1x1                              |
| BatchNorm2d        | 128 channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True                     |
| ReLU               | Activation function                                                                              |
| Conv2d             | 128 input/output channels, kernel size 3x3, stride 1x1, padding 1x1                              |
| BatchNorm2d        | 128 channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True                     |
| Projection         | Identity layer                                                                                   |
| MaxPool2d          | Kernel size 2x2, stride 2x2                                                                      |
| **Dense Layers**   | **Sequential**                                                                                   |
| Flatten            | Flatten input tensor starting from the first dimension                                           |
| Linear             | 6144 input features, 128 output features                                                         |
| ReLU               | Activation function                                                                              |
| Linear             | 128 input features, 5 output features                                                            |


### 2. 2D Transformer with Resnet block

**Architecture**:
The Transformer2DResNet model is a hybrid neural network that combines convolutional layers, residual blocks, and transformer blocks.

| Layer Type             | Details                                                                                          |
|------------------------|--------------------------------------------------------------------------------------------------|
| **Conv2D Layer**       | Conv2d(1, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))                                |
| **ResNet Block**       | **ResNetBlock2D**                                                                                |
| Conv2d                 | 128 input/output channels, kernel size 3x3, stride 1x1, padding 1x1                              |
| BatchNorm2d            | 128 channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True                     |
| ReLU                   | Activation function                                                                              |
| Conv2d                 | 128 input/output channels, kernel size 3x3, stride 1x1, padding 1x1                              |
| BatchNorm2d            | 128 channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True                     |
| Projection             | Identity layer                                                                                   |
| **Positional Encoding**| **PositionalEncoding**                                                                           |
| Dropout                | Dropout(p=0.2, inplace=False)                                                                    |
| **Transformer Blocks** | **ModuleList**                                                                                   |
| TransformerBlock       | **TransformerBlock**                                                                             |
| MultiheadAttention     | MultiheadAttention(output projection layer (in_features=128, out_features=128, bias=True))       |
| Feed-Forward           | **Sequential**                                                                                   |
| Linear                 | Linear(in_features=128, out_features=128, bias=True)                                             |
| ReLU                   | Activation function                                                                              |
| Linear                 | Linear(in_features=128, out_features=128, bias=True)                                             |

Key Components:
**Conv2D Layer**: 1 input channel and 128 output. Extracts low-level features with a 3x3 kernel, stride of 2, and 128 output channels.
**ResNet Block:** Consists of two Conv2D layers with kernel 3x3 and stride 1, each followed b batch normalization and ReLU activation. At the end a projection (identity layer), indicating no change in dimensions for the residual connection. Includes a residual connection to help with gradient flow.
**Positional Encoding**: Adds positional information to the input data. Includes a dropout layer.
**Transformer Blocks**: Multiple transformer blocks with multi-head attention and feed-forward layers.
Each block includes attention mechanisms and linear layers with ReLU activation.
**Output Layers**: A dropout layer followed by a linear layer to produce the final class predictions (5 classes).

### 3. GRU with Multihead attention 

**Architecture**:
| Layer Type             | Details                                                                                          |
|------------------------|--------------------------------------------------------------------------------------------------|
| **GRU Layer**          | GRU(1, 128, num_layers=5, batch_first=True, dropout=0.2)                                         |
| **Layer Normalization**| LayerNorm((128,), eps=1e-05, elementwise_affine=True)                                            |
| **Attention Layer**    | **MultiheadAttention**                                                                           |
| Attention              | MultiheadAttention(output projection layer (in_features=128, out_features=128, bias=True)) |
| **Attention Norm**     | LayerNorm((128,), eps=1e-05, elementwise_affine=True)                                            |
| **Linear Layer**       | Linear(in_features=128, out_features=5, bias=True)                                               |

## HYPERTUNING SEARCHSPACE
 
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
Hyperparameter tuning began with a manual exploration of parameter ranges, followed by a more focused search for optimal values. This approach helped refine the model’s performance further.
In the second stage of tuning, the process expanded to include not only the architectural parameters (e.g., number of layers, hidden sizes, and blocks) but also other important factors such as batch size, optimizer choice and scheduler types as parameter of the configuration. This holistic approach enabled more precise optimization, leading to improved training dynamics and model performance.

## RESULTS AND REFLECTIONS
Initial Hypothesis: The initial hypothesis posited that 1D models, specifically GRU and 1D CNN, would better fit the training set due to the sequential nature of the data. This was partly true, as the 1D models did perform decently on the dataset, but the performance was hindered by the dataset's imbalance. Despite attempts to address this, the 1D models overfit to the majority class.

- CNN 2D:
Results: The 2D CNN architecture emerged as the top performer for this dataset. It handled the semi-imbalanced data well, especially with the class weights applied, and delivered strong performance overall.
Reasoning: CNNs are particularly suited for spatial data and can learn hierarchical features in images or sequences and with the implementation of extra residual blocks the CNN was able to retain more information and avoid overfitting. This model was quicker to train, and deliverd the top results.

- 2D Transformer: Implementing 2D convolutions within the transformer architecture was a good idea and gave better results than 1D convolution. The 2D convolutions capture more detailed spatial features, which complements the transformer’s self-attention mechanism. Conclusion: 2D convolutions combined with transformer blocks seem to be a promising architecture for this task, however the training takes much longer to train and hypertune due to the extra parameters. Also the 

- 1D CNN:
Results: Both architectures suffered from overfitting to the majority class, especially after applying upsampling. The models essentially memorized the majority class patterns and struggled to learn features for the minority classes. The GRU was at first underperforming. I discovered that recurrent neural networks (RNNs), such as GRUs, may not be the best choice for imbalanced datasets. Their capacity to "remember" the majority class can lead to poor generalization, especially when faced with minority classes, but they are great with timeseries balanced datasets.

Balancing the Dataset: Upon balancing the dataset using oversampling techniques, 1D models still struggled. In fact, they underperformed relative to 2D CNN and Transformer models, which were surprisingly better. To enhance the performance of these models, I introduced ResNet blocks. The ResNet blocks enabled the models to focus on local features, improving generalization and helping them maintain consistent performance throughout the training. This modification improved the CNN’s ability to capture essential features, leading to better generalization.

Performance with SMOTE (Synthetic Data): When I switched to a synthetic dataset generated by SMOTE (Synthetic Minority Over-sampling Technique), the results shifted drastically. Surprisingly, the 1D models began to perform much better. This was particularly evident with the GRU, which achieved the best performance seen throughout the experiment. On the other hand, the 2D models (especially the Transformer) seemed to struggle, underperforming compared to when the real dataset was used. This suggests that Transformers may have been treating the synthetic data as noise, likely due to its inability to distinguish between real and synthetic data effectively.

Why 1D Models Excelled with SMOTE: Interestingly, 1D models (specifically the GRU) thrived with the synthetic dataset. The GRU not only outperformed previous results but also reached new heights in terms of model accuracy. It was particularly sensitive to the synthetic data, which may have allowed it to better capture underlying temporal patterns that were less apparent in the original, imbalanced dataset. Given that GRU models are good at modeling sequential relationships, the synthetic data might have provided clearer signal patterns, boosting performance.

Layer Normalization for GRU: While the GRU was achieving great performance, it was training slower than expected. To speed up the training process, I applied Layer Normalization. This adjustment allowed the GRU to converge faster, as it stabilized the learning process by normalizing the activations. This likely helped mitigate issues like vanishing gradients, leading to smoother and quicker convergence.

Other learnings and insights: I tried hypertuning optimizers, patience, batch and schedulers and found out that default values sometimes are often a good choice. Futher I learned that without a seed in the configuration it is impossible to reproduce the exact model that has been trained. 

Connect your results to your hypotheses. What did you discover? What are new insights? What hypotheses were confirmed, and which were rejected?
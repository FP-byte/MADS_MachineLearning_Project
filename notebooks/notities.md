De **F1-score**, **precision** en **recall** zijn belangrijke evaluatiemetrics voor classificatiemodellen, vooral wanneer de dataset **ongebalanceerd** is, zoals in jouw geval. Ze geven je inzicht in de prestaties van een model door te kijken naar hoe goed het model omgaat met de verschillende klassen, vooral de minder frequent voorkomende klassen.

Laten we deze metrics stap voor stap uitleggen:

### 1. **Confusion Matrix**
Voordat we dieper ingaan op de individuele metrics, is het belangrijk om de **confusion matrix** te begrijpen. Dit is een tabel die de werkelijke versus voorspelde labels weergeeft. Voor een binaire classificatie (bijvoorbeeld positieve of negatieve gevallen), ziet de confusion matrix er als volgt uit:

|                    | Voorspeld Positief (P) | Voorspeld Negatief (N) |
|--------------------|------------------------|------------------------|
| Werkelijk Positief  | True Positive (TP)      | False Negative (FN)     |
| Werkelijk Negatief  | False Positive (FP)     | True Negative (TN)      |

In jouw geval met vijf klassen (N, S, V, F, Q), wordt de confusion matrix uitgebreid naar een **multiclass matrix** waarin elke cel aangeeft hoe goed het model een bepaalde klasse voorspelt tegenover de werkelijke klasse.

### 2. **Precision**
**Precision** (ook wel **positieve voorspellingsnauwkeurigheid**) geeft aan hoeveel van de voorspellingen van de positieve klasse correct zijn. Dit is vooral belangrijk wanneer je wilt weten hoe betrouwbaar de voorspellingen van een specifieke klasse zijn.

#### Formule:
\[
\text{Precision} = \frac{TP}{TP + FP}
\]
Waar:
- **TP** = True Positives: Aantal correct voorspelde positieve gevallen.
- **FP** = False Positives: Aantal gevallen die als positief werden geclassificeerd, maar eigenlijk negatief zijn.

**Interpretatie**: Een hoge precision betekent dat wanneer het model voorspelt dat een voorbeeld van een bepaalde klasse is (bijvoorbeeld 'S' voor een specifieke aritmie), het vaak gelijk heeft. Het voorkomt dus dat je veel valse positieven hebt.

**Voorbeeld**: Als je een model hebt dat een 'S' (bijvoorbeeld 'S' voor een specifieke aritmie) voorspelt, en dit is niet correct, is dat een **false positive**. Dit betekent dat je bijvoorbeeld een ander soort aritmie hebt gemist (dat is een **false negative**). Een hoge precision zorgt ervoor dat het model minder snel zulke fouten maakt.

### 3. **Recall**
**Recall** (ook wel **gevoeligheid** of **True Positive Rate**) meet hoeveel van de werkelijke positieve gevallen correct door het model worden voorspeld. Het laat zien hoe goed je model is in het **detecteren** van een bepaalde klasse.

#### Formule:
\[
\text{Recall} = \frac{TP}{TP + FN}
\]
Waar:
- **TP** = True Positives: Aantal correct voorspelde positieve gevallen.
- **FN** = False Negatives: Aantal gevallen die als negatief werden geclassificeerd, maar eigenlijk positief zijn.

**Interpretatie**: Een hoge recall betekent dat het model goed is in het **detecteren** van gevallen van een specifieke klasse, zelfs als het een aantal fout-positieve voorspellingen maakt. Het is belangrijk voor situaties waarin je zoveel mogelijk positieve gevallen wilt identificeren, zoals in medische diagnostiek, waar je geen gevallen wilt missen (bijvoorbeeld het missen van een hartaanval).

**Voorbeeld**: In een medisch scenario waar je een aandoening moet detecteren, wil je zoveel mogelijk werkelijke gevallen (bijvoorbeeld een specifieke hartslag) detecteren, zelfs als dat betekent dat je een paar fout-positieven hebt (dat wil zeggen, het model kan verkeerd zeggen dat iemand een hartaanval heeft, hoewel dat niet het geval is).

### 4. **F1-score**
De **F1-score** is de **gewogen harmonische gemiddelde** van precision en recall. Het combineert zowel de **precision** als de **recall** in één enkele waarde, wat handig is als je zowel fout-positieven als fout-negatieven wilt minimaliseren. Dit is vooral nuttig in gevallen van ongebalanceerde datasets, waar het belangrijk is om beide aspecten goed in balans te houden.

#### Formule:
\[
\text{F1-score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

**Interpretatie**: De F1-score is een goede metric wanneer je een balans wilt tussen **precision** en **recall**, vooral wanneer de klassen ongebalanceerd zijn. Als je alleen naar precision of recall zou kijken, zou je een vertekend beeld krijgen van hoe goed je model werkt voor de minder frequent voorkomende klassen.

**Voorbeeld**: Als een model heel goed is in het voorspellen van de positieve klasse (hoge recall), maar tegelijkertijd veel fout-positieven heeft (lage precision), kan de F1-score helpen om dit evenwicht te herstellen. Een hoge F1-score betekent dat zowel precision als recall goed zijn.

### 5. **Waarom F1-score belangrijk is in ongebalanceerde datasets**
Bij ongebalanceerde datasets (zoals bij medische data, waarbij sommige aandoeningen zeldzaam zijn) is de **accuracy** (nauwkeurigheid) vaak niet een goede indicator van prestaties. Stel je voor dat je model altijd de meest voorkomende klasse voorspelt, dan zou je een hoge accuracy kunnen hebben, zelfs als het model nooit de zeldzame gevallen goed voorspelt. 

Daarom zijn **precision**, **recall**, en **F1-score** veel betere indicatoren, omdat ze je helpen te begrijpen hoe goed je model presteert op **alle klassen**, inclusief de minder vertegenwoordigde klassen.

### Samenvatting
- **Precision**: Hoeveel van de voorspellingen voor een bepaalde klasse zijn daadwerkelijk correct?
- **Recall**: Hoeveel van de werkelijke gevallen van een bepaalde klasse zijn correct voorspeld?
- **F1-score**: Het gewogen gemiddelde van precision en recall, wat zorgt voor een balans tussen de twee. Het is handig bij ongebalanceerde datasets.

Als je een model voor medische gegevens zoals hartslagclassificatie traint, wil je mogelijk dat het model niet alleen **correct** voorspelt, maar ook **alle relevante gevallen detecteert**, zodat je geen ziektes of aandoeningen mist. Daarom is het gebruik van de **F1-score** vaak zeer waardevol.

Het verschil tussen F1-score micro en F1-score macro heeft te maken met hoe de scores per klasse worden geaggregeerd (samengevoegd) om de uiteindelijke score te berekenen in een multiclass classificatie probleem.

De **micro F1-score** berekent de globale precision, recall en F1-score door alle True Positives (TP), False Positives (FP), en False Negatives (FN) over alle klassen te aggregeren, voordat de F1-score wordt berekend. Het berekent de geaggregateerde score over alle klassen heen, waardoor de score meer zegt over de algemene prestaties van het model.

De **macro F1-score** berekent de F1-score per klasse afzonderlijk en neemt vervolgens het gemiddelde van deze scores. Dit betekent dat elke klasse evenveel gewicht krijgt bij de berekening, ongeacht het aantal voorbeelden in de klasse (en dus ongeacht hoe frequent of zeldzaam de klasse is).

**Micro F1-score:**

Aggregateert de prestaties over alle klassen.
Houdt rekening met het totale aantal True Positives, False Positives, en False Negatives.
Voordeel: Het is goed als je de prestaties van het model als geheel wilt beoordelen, vooral als de klassen ongeveer dezelfde frequentie hebben.
Nadeel: Het kan vertekend worden door dominante klassen als de dataset ongebalanceerd is.

**Macro F1-score:**
Bereken de F1-score per klasse en neem vervolgens het gemiddelde van die scores.
Geeft gelijk gewicht aan elke klasse, ongeacht de frequentie van die klasse.
Voordeel: Het is nuttig wanneer je wilt dat je model evenveel aandacht besteedt aan minder vertegenwoordigde klassen.
Nadeel: Het kan gevoelig zijn voor slechte prestaties op zeldzame klassen (zoals de zeldzamere hartproblemen in je dataset), wat de score verlaagt, zelfs als de prestaties op andere klassen goed zijn.

To monitor the F1 score during training in PyTorch and adjust the learning rate based on its performance (e.g., with a scheduler like ReduceLROnPlateau), you need to compute the F1 score at the end of each epoch or batch. Here's a step-by-step guide on how to do that:
1. Install Dependencies
First, make sure you have scikit-learn installed, as it provides a convenient function to compute the F1 score. 
pip install scikit-learn
2. Compute F1 Score in PyTorch
During training, you typically use predictions and true labels to compute the F1 score. You can use scikit-learn's f1_score function to do this.

3. Monitoring F1 Score with ReduceLROnPlateau Scheduler
The ReduceLROnPlateau scheduler can be used to adjust the learning rate when the F1 score stops improving.

Here is an example of how to compute and monitor the F1 score during training and use it with ReduceLROnPlateau:
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

# Example model and optimizer
model = nn.Sequential(
    nn.Conv1d(10, 32, 3),
    nn.ReLU(),
    nn.Transformer(d_model=32, nhead=4, num_encoder_layers=6, num_decoder_layers=6)
)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data for training (e.g., a classification task)
X_train = torch.randn(100, 10, 32)  # 100 samples, 10 features, 32 timesteps
y_train = torch.randint(0, 5, (100,))  # 100 labels for 5 classes (0-4)

# DataLoader for batching
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# F1 score computation during training
def compute_f1_score(predictions, labels):
    # Assuming predictions are logits and labels are integers
    pred_labels = torch.argmax(predictions, dim=1)
    return f1_score(labels.cpu(), pred_labels.cpu(), average='macro')

# Learning rate scheduler based on F1 score
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_batch)
        loss = F.cross_entropy(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(outputs)
        all_labels.append(y_batch)
    
    # Compute F1 score after each epoch
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    epoch_f1 = compute_f1_score(all_preds, all_labels)
    
    # Step the scheduler based on F1 score
    scheduler.step(epoch_f1)
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, F1 Score: {epoch_f1:.4f}")

## F1 Score Calculation:
compute_f1_score: This function computes the F1 score using scikit-learn's f1_score. The model outputs logits, so we use torch.argmax to get the predicted class indices, and then compare them to the true labels.
average='macro': This computes the F1 score for each class independently and then averages them. You can change this depending on your specific needs (e.g., micro, weighted).
Learning Rate Scheduler: The ReduceLROnPlateau scheduler monitors the F1 score (via mode='max' since we're trying to maximize it) and reduces the learning rate if the F1 score doesn't improve after a certain number of epochs (patience).

1. Sensitivity (Recall):
Definition: The proportion of actual positive cases correctly identified by the model.
 
Why it's important:
In medical contexts, sensitivity is often the most critical metric, especially in disease detection (e.g., cancer, rare diseases).
It reflects the model's ability to correctly identify patients who actually have the condition.
False negatives (i.e., failing to identify a patient with a disease) are highly undesirable, as it can lead to a patient not receiving necessary treatment.
Example: In cancer detection, you don't want to miss diagnosing a patient who has cancer, even if this means a few false positives.
2. Specificity:
Definition: The proportion of actual negative cases correctly identified by the model.
Why it's important:
Specificity is important in ensuring that the model does not incorrectly classify healthy patients as diseased (false positives).
Example: In a test for a rare disease, a high specificity ensures that healthy patients are not subjected to unnecessary treatments or interventions.
3. Precision:
Definition: The proportion of positive predictions that are actually correct.

Why it's important:
In cases where false positives (misclassifying healthy patients as sick) can lead to unnecessary and harmful treatments, precision becomes a vital metric.
Example: In diagnosing a disease with severe side effects from treatment (e.g., chemotherapy), it's important to minimize false positives to avoid unnecessary treatments.
4. F1 Score:
 
Why it's important:
The F1 score is particularly useful when dealing with imbalanced datasets (which is common in medical datasets where one class is much rarer than the other).
It balances precision and recall, making it a good choice when both false positives and false negatives are costly.
Example: In a case of rare disease prediction, where both missing a patient (false negative) and wrongly diagnosing a healthy patient (false positive) are detrimental, the F1 score helps to optimize both.

**Which Metric Should You Focus On?**
The most important metric depends on the specific medical task and how the costs of different types of errors (false positives, false negatives) are viewed in the medical context:

For early diagnosis or screening:
If missing a positive case (false negative) is dangerous (e.g., cancer, sepsis), sensitivity or recall should be prioritized to ensure no patient with the condition is missed.
For diagnosis confirmation:
If incorrectly diagnosing a healthy person (false positive) is problematic (e.g., misdiagnosing a healthy patient as having a severe condition), then precision should be prioritized to minimize unnecessary treatments.
For imbalanced datasets (common in medical data):
The F1 score or MCC may be more informative than accuracy since they balance the trade-off between precision and recall and are not biased toward the majority class.
For model comparison:
ROC-AUC is useful to assess overall model performance across various decision thresholds, particularly in situations where you need to balance sensitivity and specificity.
For overall performance balance:
If both false positives and false negatives are costly, F1 score or MCC will give a good balance of performance.

### Example Use Case:
Heart Failure Detection:
Sensitivity (recall) would be prioritized because missing a heart disease diagnosis could be life-threatening.
However, precision (avoiding false positives) and F1 score would also be important because unnecessary treatments can be harmful.

For a medical model, sensitivity/recall is often the most critical metric, especially in early detection and screening tasks where false negatives are more dangerous. However, the F1 score and ROC-AUC are also key metrics to ensure a balanced and effective model, particularly when dealing with imbalanced classes or complex trade-offs between different types of errors. Always consider the clinical context to decide which metric best aligns with the patient’s safety and the treatment's effectiveness.


# Additionals techniques

1. Attention Mechanisms:
While you’ve already used Squeeze-and-Excitation (SE), there are other types of attention mechanisms that could complement your existing architecture, particularly within the Transformer or CNN blocks:

Self-Attention in CNN (SACN): This type of attention can be used to allow the CNN model to focus more on important spatial regions within the image (or feature map) itself. Integrating self-attention modules can refine the CNN's ability to focus on relevant spatial patterns, improving performance on more complex datasets.
Multi-Head Attention: If you're using Transformers, you could experiment with multi-head attention where each head learns different aspects of the input data. This could help the model better capture diverse features, particularly in the presence of imbalanced data where the minority classes might need different feature representations.
2. Focal Loss:
Focal Loss is an effective solution for handling class imbalance, especially in tasks where the class distribution is heavily skewed. Unlike standard cross-entropy loss, focal loss applies more weight to hard-to-classify examples, which can help the model focus on learning the minority classes.
This loss function is often combined with upsampling or downsampling techniques, as it reduces the relative loss for well-classified examples and focuses on misclassified ones, preventing the model from being overwhelmed by the majority class.
3. Mixup or CutMix:
Mixup and CutMix are data augmentation techniques that can be used to improve generalization, especially when training on imbalanced datasets.
Mixup generates new training samples by combining pairs of samples and their labels. This forces the model to learn more robust features and prevents it from focusing too much on individual classes.
CutMix takes a more aggressive approach by mixing patches from two images, blending them together, which can help the model to focus on different regions of the data, improving feature robustness.
4. Label Smoothing:
Label Smoothing is a technique that softens the target labels, reducing the model’s confidence in its predictions and improving generalization. Instead of using a hard 0 or 1 label, label smoothing replaces the true class with a value slightly less than 1, and the others with values slightly larger than 0.
This can help reduce overfitting, especially when dealing with imbalanced data, as it prevents the model from becoming overly confident about the majority class.
5. Feature Pyramid Networks (FPN):
FPNs could be helpful if you’re working with datasets that have varying scales of important features (e.g., small objects vs. large objects). This architecture is designed to create feature pyramids that allow the model to capture features at different scales, enhancing the model’s ability to generalize over different resolutions and feature sizes.
Integrating FPNs into your CNN architecture could improve its ability to recognize hierarchical features, especially if your dataset has high variability in feature sizes.
6. Adaptive Gradient Clipping (AGC):
Adaptive Gradient Clipping is a method that dynamically adjusts the gradient clipping threshold during training based on the magnitude of the gradients. It helps stabilize training, especially when dealing with highly imbalanced datasets where one class may dominate the gradients.
This technique can help prevent overfitting by ensuring the gradients don't explode or vanish during backpropagation, thus allowing for better optimization and more stable learning.
7. Few-Shot Learning Techniques:
If your dataset’s minority classes have very few examples, few-shot learning methods like prototypical networks or metric learning could be useful. These models focus on learning a similarity measure between examples, which could help improve the model’s ability to generalize to the minority classes by leveraging learned representations of class prototypes.
8. Convolutional Block Attention Module (CBAM):
CBAM is a lightweight attention module that can be applied to the convolutional feature maps. It includes both channel attention and spatial attention, refining the model’s focus on important spatial regions and channels. Adding CBAM could further refine the feature extraction process and improve the model’s sensitivity to key features, particularly in cases where you are dealing with complex patterns.
9. Semi-Supervised Learning (SSL) Techniques:
Pseudo-labelling or Consistency Regularization (used in semi-supervised learning) could help improve performance when you have a small number of labeled samples, especially in the minority class. You can train your model on the available labeled data and then use it to predict pseudo-labels for the unlabeled data, gradually increasing the amount of useful data during training.
SSL can be particularly beneficial in situations where labeled data for the minority class is scarce, and you want to leverage any available unlabeled data to improve model performance.
10. Balanced Batch Generator:
If you're using a batch generator for training, consider making it balanced by sampling batches that have an equal representation of classes. This ensures that each batch used for training has a similar class distribution, reducing bias toward the majority class during training.
This is particularly useful in combination with class weights or focal loss, as it ensures that every batch contains enough samples from minority classes to prevent the model from ignoring them.
Summary of Additional Suggestions:
Attention Mechanisms: Explore self-attention or multi-head attention in CNNs or Transformers.
Loss Functions: Implement Focal Loss or Label Smoothing to address class imbalance.
Data Augmentation: Experiment with Mixup or CutMix to improve generalization.
Architectural Enhancements: Consider adding Feature Pyramid Networks or Convolutional Block Attention Modules (CBAM) to improve feature extraction.
Gradient Stabilization: Use Adaptive Gradient Clipping (AGC) for more stable training.
Few-Shot Learning: Explore few-shot learning techniques for better handling of the minority class.
Semi-Supervised Learning: Apply pseudo-labelling or consistency regularization for improving model performance on limited labeled data.

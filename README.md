# SentimentScope: Transformer-Based Sentiment Analysis on IMDB Dataset

Final project for Udacity's Future AI Scientist Nanodegree program - Building a custom transformer model from scratch for binary sentiment classification.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Step-by-Step Guide](#step-by-step-guide)
- [Results](#results)
- [Key Takeaways](#key-takeaways)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project implements a custom transformer-based model for sentiment analysis on movie reviews.   
As a Machine Learning Engineer at Cinescope (fictional company), the goal is to build a model that can classify IMDB movie reviews as positive or negative to enhance the company's recommendation system.

### Learning Objectives
- Load, explore, and prepare text datasets for transformer models using PyTorch
- Customize transformer architecture for binary classification tasks
- Train and evaluate transformer models on sentiment analysis
- Achieve >75% accuracy on the IMDB test dataset

## Dataset

The project uses the [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/):
- **Training samples**: 25,000 reviews
- **Test samples**: 25,000 reviews
- **Classes**: Binary (Positive=1, Negative=0)
- **Format**: Text files organized in pos/neg directories

### Dataset Structure
```
aclIMDB/
├── train/
│   ├── pos/    # Positive reviews (label: 1)
│   ├── neg/    # Negative reviews (label: 0)
│   └── unsup/  # Unsupervised data (not used)
└── test/
    ├── pos/    # Positive test reviews
    └── neg/    # Negative test reviews
```

## Architecture

### Model Components

1. **AttentionHead**: Single attention head with Q, K, V projections
2. **MultiHeadAttention**: Combines multiple attention heads
3. **FeedForward**: Position-wise feed-forward network with GELU activation
4. **Block**: Complete transformer block with attention and feed-forward layers
5. **DemoGPT**: Full transformer model adapted for classification

### Model Configuration
```python
config = {
    "vocabulary_size": 30522,  # BERT tokenizer vocab size
    "num_classes": 2,          # Binary classification
    "d_embed": 128,            # Embedding dimension
    "context_size": 128,       # Maximum sequence length
    "layers_num": 4,           # Number of transformer blocks
    "heads_num": 4,            # Number of attention heads
    "head_size": 32,           # Dimension per head
    "dropout_rate": 0.1,       # Dropout probability
    "use_bias": True           # Use bias in linear layers
}
```

### Key Architecture Adaptations for Classification

1. **Mean Pooling**: Aggregate token embeddings across sequence dimension
   ```python
   x = x.mean(dim=1)  # (B, T, d_embed) → (B, d_embed)
   ```

2. **Classification Head**: Linear layer mapping to class logits
   ```python
   self.classification_head = nn.Linear(d_embed, num_classes, bias=False)
   ```

##  Installation

### Requirements
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy matplotlib
```

### Clone Repository
```bash
git clone https://github.com/yourusername/sentimentscope.git
cd sentimentscope
```

### Download Dataset
The dataset is automatically extracted in the notebook. Ensure `aclImdb_v1.tar.gz` is in the project directory.

##  Usage

### Training the Model
```python
# Initialize model
model = DemoGPT(config).to(device)

# Train for 10 epochs
EPOCHS = 10
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    # Training code here
    ...
```

### Evaluating the Model
```python
# Calculate test accuracy
test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.2f}%")
```

##  Project Structure

```
sentimentscope/
├── SentimentScope_starter.ipynb  # Main notebook
├── aclImdb_v1.tar.gz             # IMDB dataset
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

##  Step-by-Step Guide

### 1. Data Loading and Exploration

**Load Dataset Function**
```python
def load_dataset(folder):
    """Reads all text files in a folder and returns their content as a list."""
    reviews = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            reviews.append(content)
    return reviews
```

**Key Steps:**
- Load positive and negative reviews from train/test directories
- Create pandas DataFrames with review text and labels
- Split training data into train (90%) and validation (10%) sets
- Shuffle data to ensure balanced batches

**Exploration:**
- Visualize label distribution (balanced: 12,500 positive, 12,500 negative)
- Analyze review length distribution
- Sample positive and negative reviews

### 2. Tokenization

**Using BERT Tokenizer**
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**Subword Tokenization Process:**
1. Text → Subword splitting (e.g., "unhappiness" → ["un", "happiness"])
2. Subwords → Token IDs (numerical representation)
3. Add special tokens: [CLS], [SEP]
4. Apply padding/truncation to fixed length (128 tokens)

**Example:**
```
Input: "I Love Transformers."
Tokens: ['i', 'love', 'trans', '##formers', '.']
IDs: [101, 1045, 2293, ...]
```

### 3. Custom Dataset Implementation

```python
class IMDBDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        review = str(self.data.iloc[idx]['review'])
        label = self.data.iloc[idx]['label']
        
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }, label
```

**Key Points:**
- Returns dictionary of tensors (input_ids, attention_mask) and label
- Handles variable-length reviews through padding/truncation
- Uses `.iloc` for robust pandas indexing

### 4. DataLoader Setup

```python
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

**Shuffle Strategy:**
- Training: `shuffle=True` (randomize for better generalization)
- Validation/Test: `shuffle=False` (consistent evaluation)

### 5. Model Architecture Details

**AttentionHead Implementation**
```python
class AttentionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.Q_weights = nn.Linear(d_embed, head_size)
        self.K_weights = nn.Linear(d_embed, head_size)
        self.V_weights = nn.Linear(d_embed, head_size)
        # Causal mask for autoregressive attention
        
    def forward(self, input):
        Q = self.Q_weights(input)  # (B, T, head_size)
        K = self.K_weights(input)
        V = self.V_weights(input)
        
        # Scaled dot-product attention
        attention_scores = Q @ K.transpose(1, 2) / sqrt(head_size)
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = softmax(attention_scores, dim=-1)
        
        return attention_scores @ V
```

**Block Structure**
```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head = MultiHeadAttention(config)
        self.layer_norm_1 = nn.LayerNorm(d_embed)
        self.feed_forward = FeedForward(config)
        self.layer_norm_2 = nn.LayerNorm(d_embed)
    
    def forward(self, input):
        # Pre-norm architecture
        x = input + self.multi_head(self.layer_norm_1(input))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x
```

**Classification Adaptation**
```python
class DemoGPT(nn.Module):
    def forward(self, token_ids, attention_mask=None):
        # 1. Token + Position embeddings
        x = self.token_embedding_layer(token_ids)
        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        pos_embed = self.positional_embedding_layer(positions)
        x = x + pos_embed.unsqueeze(0)
        
        # 2. Transformer layers
        x = self.layers(x)  # (B, T, d_embed)
        x = self.layer_norm(x)
        
        # 3. Mean pooling for classification
        x = x.mean(dim=1)  # (B, d_embed)
        
        # 4. Classification head
        logits = self.classification_head(x)  # (B, num_classes)
        
        return logits
```

### 6. Training Process

**Training Loop Structure**
```python
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    
    for step, (inputs, labels) in enumerate(train_loader):
        # 1. Move to device
        input_ids = inputs['input_ids'].to(device)
        labels = labels.to(device)
        
        # 2. Zero gradients
        optimizer.zero_grad()
        
        # 3. Forward pass
        logits = model(input_ids)
        
        # 4. Calculate loss
        loss = criterion(logits, labels)
        
        # 5. Backward pass
        loss.backward()
        
        # 6. Update weights
        optimizer.step()
        
        running_loss += loss.item()
    
    # Validation after each epoch
    val_accuracy = calculate_accuracy(model, val_loader, device)
```

**Loss Function:**
- CrossEntropyLoss for multi-class classification
- Combines LogSoftmax and NLLLoss
- Works with raw logits (no need for manual softmax)

**Optimizer:**
- AdamW (Adam with weight decay)
- Learning rate: 3e-4
- Weight decay for regularization

### 7. Evaluation

**Accuracy Calculation**
```python
def calculate_accuracy(model, data_loader, device):
    model.eval()  # Disable dropout/batch norm
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():  # Disable gradient computation
        for batch in data_loader:
            inputs, labels = batch
            input_ids = inputs['input_ids'].to(device)
            labels = labels.to(device)
            
            # Get predictions
            logits = model(input_ids)
            predictions = torch.argmax(logits, dim=1)
            
            # Count correct predictions
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    model.train()  # Re-enable training mode
    return (total_correct / total_samples) * 100
```

**Key Points:**
- `model.eval()`: Disables dropout and batch normalization
- `torch.no_grad()`: Saves memory by not computing gradients
- `torch.argmax()`: Selects class with highest probability

##  Results

### Training Progress

| Epoch | Training Loss | Validation Accuracy |
|-------|--------------|---------------------|
| 1     | 0.5621       | 66.96%             |
| 2     | 0.4885       | 74.76%             |
| 3     | 0.4210       | 77.40%             |
| 4     | 0.3765       | 78.28%             |
| 5     | 0.3417       | 76.68%             |
| 6     | 0.2903       | 79.16%             |
| 7     | 0.2546       | 79.72%             |
| 8     | 0.2015       | 79.44%             |
| 9     | 0.1672       | 79.88%             |
| 10    | 0.1496       | 79.32%             |

### Final Performance

- **Initial Accuracy** (untrained): 49.88% (random baseline)
- **Final Validation Accuracy**: 79.32%
- **Test Accuracy**: **76.82%** ✅ (Goal: >75%)

### Performance Analysis

**Learning Curve:**
- Rapid improvement in first 3 epochs (66% → 77%)
- Gradual refinement epochs 4-10 (77% → 79%)
- Loss decreased from 0.69 to 0.15
- Some validation accuracy fluctuation (76.68% at epoch 5) indicates slight overfitting

**Model Behavior:**
- Consistent improvement in training loss
- Validation accuracy plateaued around 79%
- Test accuracy (76.82%) slightly lower than validation (79.32%), suggesting good generalization
- Successfully exceeded the 75% target accuracy

## 💡 Key Takeaways

### 1. Transformer Adaptation for Classification
- **Challenge**: Standard transformers output sequences (B, T, d_embed), but classification needs single vectors (B, num_classes)
- **Solution**: Mean pooling across sequence dimension before classification head
- **Impact**: Enables using powerful transformer architectures for classification tasks

### 2. Robust Data Pipeline
- **Custom Dataset**: Implemented proper handling of pandas DataFrames with `.iloc` indexing
- **Tokenization**: BERT's subword tokenization handles out-of-vocabulary words effectively
- **Batching**: DataLoader with proper padding/truncation ensures uniform input shapes

### 3. Hyperparameter Effectiveness
- **Learning Rate** (3e-4): Balanced convergence speed and stability
- **Epochs** (10): Sufficient for convergence without excessive overfitting
- **Batch Size** (32): Good trade-off between memory and gradient stability
- **Model Size**: 4 layers with 128-dim embeddings provided good capacity without overfitting

### 4. Training Insights
- **Optimizer**: AdamW with weight decay prevented overfitting
- **Loss Function**: CrossEntropyLoss appropriate for classification
- **Evaluation Mode**: Critical to disable dropout during evaluation
- **Gradient Management**: `zero_grad()` before each batch prevents gradient accumulation

##  Future Improvements

### Model Enhancements
1. **Increase Model Capacity**
   - More layers (6-12 blocks)
   - Larger embeddings (256-512 dimensions)
   - More attention heads (8-16 heads)

2. **Advanced Techniques**
   - Learning rate scheduling (cosine annealing)
   - Gradient clipping for stability
   - Mixed precision training (FP16)
   - Data augmentation (back-translation, synonym replacement)

3. **Architecture Variations**
   - Try different pooling strategies (max pooling, CLS token)
   - Experiment with pre-norm vs post-norm
   - Add dropout layers strategically

### Training Improvements
1. **Extended Training**
   - Train for 20-30 epochs with early stopping
   - Use validation loss for early stopping criterion

2. **Hyperparameter Tuning**
   - Grid search or Bayesian optimization
   - Try different learning rates (1e-4, 5e-4)
   - Experiment with weight decay values

3. **Transfer Learning**
   - Fine-tune pre-trained BERT/RoBERTa
   - Compare custom model vs pre-trained performance

### Evaluation Enhancements
1. **Additional Metrics**
   - Precision, Recall, F1-score
   - Confusion matrix analysis
   - Per-class performance

2. **Error Analysis**
   - Analyze misclassified examples
   - Identify common failure patterns
   - Review attention weights

##  Key Concepts Explained

### Attention Mechanism
- **Purpose**: Allows model to focus on relevant parts of input
- **Process**: Query-Key-Value system computes weighted sum
- **Benefit**: Captures long-range dependencies in text

### Tokenization
- **Subword**: Splits words into meaningful pieces
- **Advantages**: Handles rare words, reduces vocabulary size
- **Example**: "unhappiness" → ["un", "##happiness"]

### Mean Pooling
- **Purpose**: Convert sequence embeddings to single vector
- **Method**: Average all token embeddings along sequence dimension
- **Alternative**: Use [CLS] token or max pooling

### Cross-Entropy Loss
- **Use Case**: Multi-class classification (even for binary: 2 classes)
- **Formula**: Measures difference between predicted and true distributions
- **Implementation**: Combines LogSoftmax + NLLLoss in PyTorch

##  Acknowledgments

- **Udacity** - Future AI Scientist Nanodegree Program
- **Stanford AI Lab** - IMDB Dataset
- **Hugging Face** - Transformers Library and BERT Tokenizer
- **PyTorch Team** - Deep Learning Framework

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

⭐ If you found this project helpful, please give it a star!

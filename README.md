# Vision Transformer (ViT) — From Scratch  
This repository provides a complete from-scratch implementation of the Vision Transformer (ViT) model using PyTorch, trained and evaluated on the MNIST dataset. It demonstrates how ViT can replace convolutional networks with transformer-based architectures for image classification. Every component — from patch embedding and multi-head self-attention to the final classification head — has been manually implemented without relying on pre-built transformer APIs.

## Overview  
The project shows how to build, train, and evaluate a Vision Transformer that classifies handwritten digits (0–9) from the MNIST dataset. It includes both a Jupyter Notebook for step-by-step understanding and a Python script for standalone execution. The ViT architecture diagram is also included for visualization.

## Files  
- `vit_from_scratch.ipynb` – Jupyter Notebook version with training and explanations  
- `vit_architecture/1.png` – Vision Transformer architecture diagram

## Vision Transformer Architecture  
The Vision Transformer divides an image into small patches, embeds them, and processes them as a sequence — just like words in NLP transformers. Each patch acts as a token that participates in the self-attention mechanism, enabling the model to learn spatial relationships across the image.  

![ViT Architecture](vit_architecture/1.png)

### Core Components  
- **Patch Embedding:** Uses a convolutional layer to divide the image into patches and convert them into embeddings.  
- **Positional Embedding:** Adds positional information to each patch embedding to preserve spatial order.  
- **Transformer Encoder:** A stack of multi-head self-attention and feedforward layers with layer normalization and residual connections.  
- **CLS Token:** A learnable token prepended to represent the overall image during classification.  
- **MLP Head:** A fully connected layer that performs the final classification task.  

## Model Configuration  
- Image Size: 28×28  
- Patch Size: 7×7  
- Embedding Dimension: 16  
- Attention Heads: 4  
- Transformer Blocks: 4  
- Hidden Nodes (MLP): 64  
- Number of Classes: 10 (digits 0–9)  
- Optimizer: Adam  
- Learning Rate: 0.001  
- Epochs: 10  

## Training Details  
The implementation uses the MNIST dataset with `CrossEntropyLoss` as the loss function and the Adam optimizer. The model supports both CPU and GPU training. During each epoch, it reports batch-wise loss and accuracy, along with an epoch summary. After training, validation accuracy is computed — typically achieving around **95% accuracy** on MNIST after 10 epochs.

## How to Run  
**Option 1 — Jupyter Notebook**  
```bash
git clone https://github.com/<your-username>/vit_from_scratch.git
cd vit_from_scratch
jupyter notebook vit_from_scratch.ipynb

## How to Run  
- Understand the core principles of Vision Transformers and how they differ from traditional CNNs.  
- Gain hands-on experience implementing self-attention, positional embeddings, and normalization layers from scratch in PyTorch.  
- Observe and analyze how ViT performs on small-scale image datasets like MNIST, highlighting its efficiency and adaptability.

## Future Improvements  
- Extend support to more complex datasets such as CIFAR-10 or ImageNet subsets.  
- Add dropout and stochastic depth regularization.  
- Implement visualization of attention maps for interpretability.  
- Integrate pretrained weights and fine-tuning options for transfer learning.  

## References  
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)  
- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [Vision Transformer tutorial — YouTube (starts ~1h 40m)](https://www.youtube.com/watch?v=DdsVwTodycw&t=6025s)





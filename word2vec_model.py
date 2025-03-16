import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import re
from tqdm import tqdm

# Sample text data
text = """ Word2Vec is a technique used in natural language processing
to represent words as vectors. It captures semantic relationships between words."""

#Function to clean and tokenize text
def preprocess(text):
    text = text.lower() #Convert to Lowercase
    text = re.sub(r'[^a-z\s]', '', text) # Remove special characters
    words = text.split() # Tokenize (split into words)
    return words

# Tokenize the text
words = preprocess(text)
print("Tokenized Words:", words)

#Build a vocabulary
word_counts = Counter(words) # Count word frequencies
vocab = list(word_counts.keys()) # Unique words
word_to_idx = {word: i for i, word in enumerate(vocab)} # Word to index mapping
idx_to_word = {i: word for word, i in word_to_idx.items()} # Index to word mapping

print("Vocabulary:", word_to_idx)

# Define context window size
window_size = 2 # Words before and after the target word

# Generate training data (target, context) pairs
training_data = []
for i, target_word in enumerate(words):
    target_idx = word_to_idx[target_word]
    for j in range(-window_size, window_size + 1):
        if j != 0 and (i + j) >= 0 and (i + j) < len(words):
            context_word = words[i + j]
            context_idx = word_to_idx[context_word]
            training_data.append((target_idx, context_idx))

print("Sample Training Data:", training_data[:5]) # Print first 5 training pairs

# Define Word2Vec model (CBOW)
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs) # Convert input words to embeddings
        out = self.linear(embeds) # Fully connected layer
        return out

# Set hyperparameters
embedding_dim = 10
vocab_size = len(vocab)

# Initialize model
model = Word2Vec(vocab_size, embedding_dim)

print(model)

# Training setup
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert training data to tensors
data = [(torch.tensor(context_words, dtype=torch.long), torch.tensor(target, dtype=torch.long)) for context_words, target in training_data]

# Training loop
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for context, target in data:
        optimizer.zero_grad()  # Reset gradients
        output = model(context)  # Forward pass
        loss = loss_function(output, target)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

print("Training completed!")

import torch.nn.functional as F

word_to_ix = {word: i for i, word in enumerate(vocab)}

def get_word_embedding(word):
    idx = word_to_ix.get(word, None)
    if idx is not None:
        return model.embeddings(torch.tensor([idx], dtype=torch.long)).detach()
    else:
        print(f"Word '{word}' not found in vocabulary.")
        return None

def find_similar_words(word, top_n=3):
    embedding = get_word_embedding(word)
    if embedding is None:
        return
    
    similarities = {}
    for w, idx in word_to_ix.items():
        if w == word:
            continue
        other_embedding = model.embeddings(torch.tensor([idx], dtype=torch.long)).detach()
        similarity = F.cosine_similarity(embedding, other_embedding).item()
        similarities[w] = similarity
    
    sorted_similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"Top {top_n} words similar to '{word}':")
    for similar_word, score in sorted_similar_words:
        print(f"{similar_word}: {score:.4f}")

# Example: Find words similar to "processing"
find_similar_words("processing")

import random

CONTEXT_SIZE = 2  # You can change this

def generate_training_data(words, word_to_ix, context_size=CONTEXT_SIZE):
    training_data = []
    for i, word in enumerate(words):
        target_word = word_to_ix[word]
        context = []
        
        # Select words before and after the target word
        for j in range(-context_size, context_size + 1):
            if j == 0 or i + j < 0 or i + j >= len(words):
                continue
            context.append(word_to_ix[words[i + j]])
        
        for context_word in context:
            training_data.append((target_word, context_word))
    
    return training_data

training_data = generate_training_data(words, word_to_ix)
print("Sample Training Data:", training_data[:10])  # Show first 10 pairs

import torch.nn as nn

# Define Word2Vec Model
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_word):
        embed = self.embeddings(input_word)
        out = self.linear(embed)
        return out

# Model parameters
EMBEDDING_DIM = 10  # You can change this to experiment
VOCAB_SIZE = len(word_to_ix)

# Initialize model, loss function, and optimizer
model = Word2Vec(VOCAB_SIZE, EMBEDDING_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Convert training data into tensor format
# Convert training data into tensor format correctly
train_data = [
    (torch.tensor(word_to_ix[list(word_to_ix.keys())[context.item()]], dtype=torch.long),
     torch.tensor(word_to_ix[list(word_to_ix.keys())[target.item()]], dtype=torch.long)) 
    for context, target in data
]

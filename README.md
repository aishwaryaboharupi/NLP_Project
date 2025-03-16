Word2Vec Implementation

This repository contains an implementation of the Word2Vec model using PyTorch. The model learns word embeddings by capturing semantic relationships between words.

FEATURES
- Tokenization and vocabulary creation
- Word2Vec model with embedding and linear layers
- Training using Skip-gram approach
- Word similarity detection

INSTALLATION

- Clone the repository:
  git clone https://github.com/aishwaryaboharupi/NLP_Project.git
  cd NLP_Project

- Create a virtual environment (optional but recommended):
  python -m venv nlp_env
  nlp_env\Scripts\activate     ( On Windows )
  
- Install dependencies:
  pip install -r requirements.txt

  
USAGE
- Run the Word2Vec training script:
  python word2vec_model.py

RESULTS
- The model trains for 50 epochs.
- Achieved loss: ~108.95
- Example similar words to 'processing':
    words: 0.4024
    language: 0.3580
    in: 0.3070

REFERENCES
- Paper Name: Efficient Estimation of Word Representations in Vector Space
- Link: https://arxiv.org/abs/1301.3781

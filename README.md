# 1st. Semester Project: AI Sentence classifier
Python project for the first semester of college: an AI sentence classifier made from scratch using NumPy.

# Features
- Simple Tokenization: Splits text into words, removes unwanted characters, and further breaks words into smaller chunks.
- Co-occurrence Matrix: Captures word context relationships within a given window size.
- PMI Calculation: Measures statistical association between words.
- SVD for Dimensionality Reduction: Converts high-dimensional word relationships into compact embedding representations.
- Embedding Lookup: Converts sentences into a sequence of embeddings.

# Prerequisites
- Python 3.9.13
- Required Packages:
  - numpy 2.0.2
  - tqdm 4.67.1

# Usage
## 1. Loading or Creating
Program will ask you:
- `'Do you want load pre-existing EmbeddingData and word dataset ? (Y/N):'` If you respond with N it will generate embeddings and create `embeddings_database.json` file. 
**WARNING:** If you say Y it will try to load `embeddings_database.json` which I don't include so it will cause an error.

- `'Do you want load pre-existing weights and biases thus skip training ? (Y/N):'` If you respond with N it will calculate the model and create `model_data.json` file.
**WARNING:** If you say Y it will try to load `model_data.json` which I don't include so it will cause an error.

## 2. Using The Model
Program will ask you `'Input sentence in polish:'`. The sentene must be either one of these types:
-  Temperature
-  Time of the day
-  Date
-  Name
-  Surname
-  Hobby
-  Food
-  Movie
-  Book
-  Animal

When passed `która godzina we wrocławiu` as input, program will return:
```
Number of tokens: 4
Total token length: 23
Average token length: 5.75
Ceiling of average token length: 6
Complexity Score (Standard Deviation of Token Lengths): 2.59

która godzina we wrocławiu: You asked about the time of the day
```
**WARNING:** In case one of the words in the inputted sentece is out of the vocabulary, the program will ask you again to pass a sentence.

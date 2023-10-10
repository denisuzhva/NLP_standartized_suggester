# NLP suggester of standartized terms

Given a text input and a list of standartized terms, suggests terms with similar meaning to the ones found in the text.

## Setup

In your environment, run 
```
pip install -r requirements.txt
```
The software requires a couple of general-purpose NLP libraries (text preprocessing, NLTK, BERT), which are installed from the requirements file.

## Use

Run

```
python FindSim.py
```
to test the software on a sample text input `./data/sample_text.txt` with a sample list of standartized terms `./data/standartized_terms.csv`.
Top suggestions for each sentence with corresponding highest similarity scores in the form `SUGGESTION -> SIMILARITY VALUE` are proposed in the output for each sentence of the text input, under each sentence.

You may wish to process your own text inputs with differing standartized terms sets.
To achieve this, run 
```
python FindSim.py --terms_path=TERMS_PATH --text_path=TEXT_PATH
```
Note that the terms should be separated by line breaks in the corresponding file.

You may also wish to get the similarity scores for all available standartized terms.
To acieve this, add `-v` or `--verbose` flag, e.g.,
```
python FindSim.py -v
```

## Architecture

The idea is to compare each sentence of the input text with each term of the desired terms list.
In this software, comparison is expressed with cosine simialrities between sentence embeddings, derived by a sentence transformer (BERT). 
The algorithm is as follows:

1. Preprocess the input text
    1. Tokenize sentences
    2. Apply transformations to each sentence (make lowercase, lemmatize, expand, etc.)
2. Preprocess the terms
    1. Apply transformations to each term string (make lowercase, lemmatize, expand, etc.)
3. Compute sentence embeddings by a sentence transformer for each sentence in the input text and each term
4. Compute (cosine) similarity between all sentences and all terms
5. Sort the terms by corresponding similarity values for each sentence

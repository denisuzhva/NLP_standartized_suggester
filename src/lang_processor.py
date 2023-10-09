import string
import re
import contractions
from itertools import count

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup
from textblob import TextBlob
from unidecode import unidecode


def lemmatize_pos_tagged_text(text, lemmatizer, pos_tag_dict):
    """
    Lemmatizes the words in the given text based on their part-of-speech tags.

    Parameters:
        text (str): The input text to be lemmatized.
        lemmatizer (nltk.stem.WordNetLemmatizer): The lemmatizer object to be used.
        pos_tag_dict (dict): A dictionary mapping NLTK part-of-speech tags to WordNet part-of-speech tags.

    Returns:
        str: The lemmatized text.
    """
    sentences = nltk.sent_tokenize(text)
    new_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        new_sentence_words = []  # one pos_tuple for sentence
        pos_tuples = nltk.pos_tag(nltk.word_tokenize(sentence))

        for word_idx, word in enumerate(nltk.word_tokenize(sentence)):
            nltk_word_pos = pos_tuples[word_idx][1]
            wordnet_word_pos = pos_tag_dict.get(nltk_word_pos[0].upper(), None)
            if wordnet_word_pos is not None:
                new_word = lemmatizer.lemmatize(word, wordnet_word_pos)
            else:
                new_word = lemmatizer.lemmatize(word)

            new_sentence_words.append(new_word)

        new_sentence = " ".join(new_sentence_words)
        new_sentences.append(new_sentence)

    return " ".join(new_sentences)


def download_if_non_existent(res_path, res_name):
    try:
        nltk.data.find(res_path)
    except LookupError:
        print(f'resource {res_path} not found. Downloading now...')
        nltk.download(res_name)


class NltkPreprocessingSteps:
    _ids = count(0)

    def __init__(self, X):
        self.id = next(self._ids)
        self.X = X
        if self.id == 0:
            download_if_non_existent('corpora/stopwords', 'stopwords')
            download_if_non_existent('tokenizers/punkt', 'punkt')
            download_if_non_existent(
                'taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
            download_if_non_existent('corpora/wordnet', 'wordnet')
            download_if_non_existent('corpora/omw-1.4', 'omw-1.4')

        self.sw_nltk = stopwords.words('english')
        self.sw_nltk.extend(['<*>'])
        self.sw_nltk.remove('not')

        self.post_tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV
        }

        self.remove_punctuations = string.punctuation.replace('.', '')

    def remove_html_tags(self):
        self.X = BeautifulSoup(self.X, 'html.parser').get_text()
        return self

    def replace_diacritics(self):
        self.X = unidecode(self.X, errors="preserve")
        return self

    def to_lower(self):
        self.X = self.X.lower()
        return self

    def expand_contractions(self):
        self.X = " ".join([contractions.fix(expanded_word)
                          for expanded_word in self.X.split()])
        return self

    def remove_numbers(self):
        self.X = re.sub(r'\d+', '', self.X)
        return self

    def replace_dots_with_spaces(self):
        self.X = re.sub("[.]", " ", self.X)
        return self

    def remove_punctuations_except_periods(self):
        self.X = re.sub('[%s]' %
                        re.escape(self.remove_punctuations), '', self.X)
        return self

    def remove_all_punctuations(self):
        self.X = re.sub('[%s]' %
                        re.escape(string.punctuation), '', self.X)
        return self

    def remove_double_spaces(self):
        self.X = re.sub(' +', ' ', self.X)
        return self

    def fix_typos(self):
        self.X = str(TextBlob(self.X).correct())
        return self

    def remove_stopwords(self):
        # remove stop words from token list in each column
        self.X = " ".join([word for word in self.X.split()
                           if word not in self.sw_nltk])
        return self

    def lemmatize(self):
        lemmatizer = WordNetLemmatizer()
        self.X = lemmatize_pos_tagged_text(
            self.X, lemmatizer, self.post_tag_dict)
        return self

    def get_processed_text(self):
        return self.X

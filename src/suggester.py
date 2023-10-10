import operator
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

from src.text_processor import NltkPreprocessingSteps


def terms_text_processor(terms, text):
    text_tokens = sent_tokenize(text)
    text_preproc = NltkPreprocessingSteps(text_tokens)
    processed_text = \
        text_preproc \
        .remove_html_tags()\
        .replace_diacritics()\
        .expand_contractions()\
        .remove_numbers()\
        .fix_typos()\
        .remove_punctuations_except_periods()\
        .lemmatize()\
        .remove_double_spaces()\
        .remove_all_punctuations()\
        .remove_stopwords()\
        .get_processed_text()
    terms_preproc = NltkPreprocessingSteps(terms)
    processed_terms = \
        terms_preproc \
        .remove_html_tags()\
        .replace_diacritics()\
        .expand_contractions()\
        .remove_numbers()\
        .fix_typos()\
        .remove_punctuations_except_periods()\
        .lemmatize()\
        .remove_double_spaces()\
        .remove_all_punctuations()\
        .remove_stopwords()\
        .get_processed_text()
    return text_tokens, processed_terms, processed_text


def get_suggestions(terms, text, sent_model='all-MiniLM-L6-v2'):
    text_tokens, pp_terms, pp_text = terms_text_processor(terms, text)
    model = SentenceTransformer(sent_model)
    scores = {}
    for pp_sent, sent in zip(pp_text, text_tokens):
        sent_emb = model.encode(pp_sent, convert_to_tensor=True)
        term_emb = model.encode(pp_terms, convert_to_tensor=True)
        cosine_scores = util.cos_sim(sent_emb, term_emb)
        scores_per_sent = sorted(
            zip(terms, cosine_scores[0].tolist()), key=lambda x: x[1], reverse=True)
        scores[sent] = scores_per_sent
    return scores

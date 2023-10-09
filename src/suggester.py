from src.lang_processor import NltkPreprocessingSteps


def terms_text_processor(terms, text):
    text_preproc = NltkPreprocessingSteps(text)
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
    terms_preproc = [NltkPreprocessingSteps(term) for term in terms]
    processed_terms = [
        term
        .remove_html_tags()
        .replace_diacritics()
        .expand_contractions()
        .remove_numbers()
        .fix_typos()
        .remove_punctuations_except_periods()
        .lemmatize()
        .remove_double_spaces()
        .remove_all_punctuations()
        .remove_stopwords()
        .get_processed_text()
        for term in terms_preproc
    ]
    return processed_terms, processed_text


def get_suggestions(terms, text):
    pp_terms, pp_text = terms_text_processor(terms, text)
    # SOME SIMILARITY MAGICK
    return None, None

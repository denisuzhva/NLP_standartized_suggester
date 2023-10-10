import argparse
from pprint import pprint
from src.suggester import get_suggestions


parser = argparse.ArgumentParser(
    description='Get paths to the standartized terms and a sample text to improve.')
parser.add_argument('--terms_path', type=str,
                    default='./data/standartized_terms.csv', help='Path to the standartized terms')
parser.add_argument('--text_path', type=str,
                    default='./data/sample_text.txt', help='Path to the sample text')
parser.add_argument('-v', '--verbose', action='store_true', default=False)
args = parser.parse_args()


def score_printer(scores, verbose=True):
    if verbose:
        for sent, scores_per_sent in scores.items():
            print(f'{sent}\n')
            for term, score in scores_per_sent:
                print(f'\t{term} -> {score}\n')
    else:
        for sent, scores_per_sent in scores.items():
            print(f'{sent}\n')
            print(
                f'\t{scores_per_sent[0][0]} -> {scores_per_sent[0][1]}\n')


if __name__ == '__main__':
    terms_path = args.terms_path
    text_path = args.text_path
    with open(terms_path, 'r') as f:
        terms = f.read().split('\n')
    with open(text_path, 'r') as f:
        text = f.read()
    scores = get_suggestions(terms, text)

    score_printer(scores, verbose=args.verbose)

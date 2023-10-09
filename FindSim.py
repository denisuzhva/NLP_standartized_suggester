import argparse
from src.suggester import get_suggestions


parser = argparse.ArgumentParser(
    description='Get paths to the standartized terms and a sample text to improve.')
parser.add_argument('--terms_path', type=str,
                    default='./data/standartized_terms.csv', help='Path to the standartized terms')
parser.add_argument('--text_path', type=str,
                    default='./data/sample_text.txt', help='Path to the sample text')
args = parser.parse_args()


if __name__ == '__main__':
    terms_path = args.terms_path
    text_path = args.text_path
    with open(terms_path, 'r') as f:
        terms = f.read().split('\n')
    with open(text_path, 'r') as f:
        text = f.read()
    pp_terms, pp_text = get_suggestions(terms, text)
    print(pp_text)
    print(pp_terms)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stopwords_en = set(stopwords.words('english'))

__tokenization_pattern = r'''(?x)          # set flag to allow verbose regexps
        \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ],
    '''
tokenizer = nltk.tokenize.regexp.RegexpTokenizer(__tokenization_pattern)


def preprocessor(text):
    stems = []
    tokens = tokenizer.tokenize(text.lower())
    for token in tokens:
        if token.isalpha() and token not in stopwords_en:
            stems.append(str(stemmer.stem(token)))
    return stems

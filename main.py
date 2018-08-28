import MySQLdb
import os
from subprocess import call
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import random

stemmer = SnowballStemmer("english")
stopwords_en = set(stopwords.words('english'))

__tokenization_pattern = r'''(?x)          # set flag to allow verbose regexps
        \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
      | (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
      | \w+(?:-\w+)*        # words with optional internal hyphens
      | \.\.\.              # ellipsis
      | [][.,;"'?():_`-]    # these are separate tokens; includes ]
    '''
tokenizer = nltk.tokenize.regexp.RegexpTokenizer(__tokenization_pattern)


def preprocessor(text):
    stems = []
    tokens = tokenizer.tokenize(text.lower())
    for token in tokens:
        if token.isalpha() and token not in stopwords_en:
            stems.append(str(stemmer.stem(token)))
    return stems


bow_vectorizer = CountVectorizer(lowercase=False,
                                 tokenizer=lambda x: x,  # because we already have tokens available
                                 stop_words=None,  # stop words removal already done from NLTK
                                 max_features=5000,  # pick top 5K words by frequency
                                 ngram_range=(1, 1),  # we want unigrams now
                                 binary=False)  # we want as binary/boolean features

token = []
name = []
x = list()
y = list()
num_test = 0
num_positive = 0
num_negative = 0

text_path_base = './text/'
pdf_path_base = '../article/'
text_names = os.listdir(text_path_base)
pdf_names = os.listdir(pdf_path_base)

db = MySQLdb.connect("localhost", "root", "lygame218", "NLP", charset='utf8')
cursor = db.cursor()
sql = "select * from NLP_ARTICLE"
try:
    cursor.execute(sql)
    results = cursor.fetchall()
except:
    print('Something Wrong!')
    db.rollback()

for row in results:
    id = row[1]
    pdf_name = row[5]
    label = row[8]
    text_name = str(id) + '.txt'
    if pdf_name not in pdf_names:
        continue
    if text_name not in text_names:
        call(["python", "pdfminer.six-master/tools/pdf2txt.py", "-o",
              text_path_base + text_name, pdf_path_base + pdf_name])
    with open(text_path_base + text_name, "r", encoding="utf-8") as f:
        text = f.read().replace(u'\xa0', ' ').replace('\n', ' ')
        if len(text) == 0:
            print('Empty Text')
            continue
        token.append(preprocessor(text))
    name.append(id)
    y.append(label)
    if label == 0:
        num_test += 1
    elif label == 1:
        num_positive += 1
    elif label == 2:
        num_negative += 1

name = np.asarray(name)
print(num_test, num_positive, num_negative)
num = num_test + num_positive + num_negative
text_vec = bow_vectorizer.fit_transform(token)
le = LabelEncoder()
train_x = []
train_y = []
test_x = []
msk = np.asarray(y) != 0
le = LabelEncoder()

train_x = text_vec[msk]
test_x = text_vec[~msk]
train_id = name[msk]
test_id = name[~msk]

y = le.fit_transform(y)
train_y = y[msk]
classifier = MultinomialNB()
classifier.fit(train_x, train_y)
preds_bow = classifier.predict(test_x)
print(le.inverse_transform(pred) for pred in preds_bow)
p = classifier.predict_proba(test_x)


for i in range(num_test):
    temp = max(p[i][0] - random.random() * 0.2, 0)
    sql = "UPDATE NLP_ARTICLE SET likelyhood = '" + str(temp) + "' WHERE articleID = '" + str(test_id[i]) + "'"
    cursor.execute(sql)
db.commit()
db.close()

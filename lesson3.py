
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# One-time download for this environment
nltk.download("wordnet")

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["playing", "studies", "running", "cars"]

for word in words:
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word, pos="v")

    print("Word:", word)
    print("Stem:", stem)
    print("Lemma:", lemma)
    print()

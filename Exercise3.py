import nltk

from nltk.corpus import gutenberg
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as pltmoby_dick = gutenberg.raw('melville-moby_dick.txt')

tokens = word_tokenize(pltmoby_dick)

stop_words = set(stopwords.words("english"))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]

pos_tags = nltk.pos_tag(filtered_tokens)

pos_freq = FreqDist(tag for (word, tag) in pos_tags)
top_pos = pos_freq.most_common(5)
print("Top 5 most common parts of speech:")
for tag, count in top_pos:
 print(tag, ":", count)

lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word, pos=tag[0].lower()) for word, tag in pos_tags[:20]]

pos_freq.plot(30, cumulative=False)
pltmoby_dick.show()
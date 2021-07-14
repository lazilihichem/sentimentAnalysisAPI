from __future__ import unicode_literals
import re, string
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import pyarabic.araby as araby
import pyarabic.trans as trans
import qalsadi.lemmatizer
import nltk
from textblob import TextBlob



def stem_english(text):
    nltk.download('wordnet')
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    zen = TextBlob(text)
    words = zen.words
    cleaned = list()
    for w in words:
        cleaned.append(lemmatizer.lemmatize(w))
    return " ".join(cleaned)


def remove_english_stop_words(text):
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stops = set(stopwords.words("english"))
    zen = TextBlob(text)
    words = zen.words
    return " ".join([w for w in words if not w in stops and len(w) >= 2])

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    return text


def split_hashtag_to_words(tag):
    tag = tag.replace('#', '')
    tags = tag.split('_')
    if len(tags) > 1:
        return tags
    pattern = re.compile(r"[A-Z][a-z]+|\d+|[A-Z]+(?![a-z])")
    return pattern.findall(tag)


def clean_hashtag(text):
    words = text.split()
    text = list()
    for word in words:
        if is_hashtag(word):
            text.extend(extract_hashtag(word))
        else:
            text.append(word)
    return " ".join(text)


def is_hashtag(word):
    if word.startswith("#"):
        return True
    else:
        return False


def extract_hashtag(text):
    hash_list = ([re.sub(r"(\W+)$", "", i) for i in text.split() if i.startswith("#")])
    word_list = []
    for word in hash_list:
        word_list.extend(split_hashtag_to_words(word))
    return word_list

def convert_emojis_to_word(text):
    try:
        import cPickle as pickle
    except ImportError:
        import pickle
    import re

    with open('api/Preprocess/Emoji_Dict.p', 'rb') as fp:
        Emoji_Dict = pickle.load(fp)
    Emoji_Dict = {v: k for k, v in Emoji_Dict.items()}
    for emot in Emoji_Dict:
        text = re.sub(r'('+emot+')', " ".join(Emoji_Dict[emot].replace(",","").replace(":","").split()), text)
        text = re.sub("_" , " " , text)
    return text


def clean_tweet(text):
    text = re.sub('#\d+K\d+', ' ', text)  # years like 2K19
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('@[^\s]+',' ',text)
    text = clean_hashtag(text)
    #text = remove_emoji(text)
    return text


def clean_english_text(text):
    ## Clean for tweets
    text = clean_tweet(text)
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)  # remove punctuation
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    ## convert emojis to word
    text = convert_emojis_to_word(text)
    ## Remove the rest of Emojis
    text = remove_emoji(text)
    ## Remove stop words
    text = remove_english_stop_words(text)
    ## Remove numbers
    text = re.sub("\d+", " ", text)
    # Stemming
    text = stem_english(text)
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)

    return text
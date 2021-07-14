from __future__ import unicode_literals
import re, string
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
import pyarabic.araby as araby
import pyarabic.trans as trans
import qalsadi.lemmatizer
import nltk
from textblob import TextBlob
def stem(text):
    zen = TextBlob(text)
    words = zen.words
    cleaned = list()
    lemmer = qalsadi.lemmatizer.Lemmatizer()
    for w in words:
        cleaned.append(lemmer.lemmatize(w))
    return " ".join(cleaned)


def normalizeArabic(text):
    text = text.strip()
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    noise = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    text = re.sub(noise, '', text)
    text = re.sub(r'(.)\1+', r"\1\1", text)  # Remove longation
    text = araby.strip_tatweel(text)
    text = araby.normalize_ligature(text)
    text = trans.normalize_digits(text, source='all', out='west')
    return araby.strip_tashkeel(text)


def remove_stop_words(text):
    zen = TextBlob(text)
    words = zen.words
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stops = set(stopwords.words("arabic"))
    stop_word_comp = {"،", "آض", "آمينَ", "آه", "آهاً", "آي", "أ", "أب", "أجل", "أجمع", "أخ", "أخذ", "أصبح", "أضحى",
                      "أقبل", "أقل", "أكثر", "ألا", "أم", "أما", "أمامك", "أمامكَ", "أمسى", "أمّا", "أن", "أنا", "أنت",
                      "أنتم", "أنتما", "أنتن", "أنتِ", "أنشأ", "أنّى", "أو", "أوشك", "أولئك", "أولئكم", "أولاء",
                      "أولالك", "أوّهْ", "أي", "أيا", "أين", "أينما", "أيّ", "أَنَّ", "أََيُّ", "أُفٍّ", "إذ", "إذا",
                      "إذاً", "إذما", "إذن", "إلى", "إليكم", "إليكما", "إليكنّ", "إليكَ", "إلَيْكَ", "إلّا", "إمّا",
                      "إن", "إنّما", "إي", "إياك", "إياكم", "إياكما", "إياكن", "إيانا", "إياه", "إياها", "إياهم",
                      "إياهما", "إياهن", "إياي", "إيهٍ", "إِنَّ", "ا", "ابتدأ", "اثر", "اجل", "احد", "اخرى", "اخلولق",
                      "اذا", "اربعة", "ارتدّ", "استحال", "اطار", "اعادة", "اعلنت", "اف", "اكثر", "اكد", "الألاء",
                      "الألى", "الا", "الاخيرة", "الان", "الاول", "الاولى", "التى", "التي", "الثاني", "الثانية",
                      "الذاتي", "الذى", "الذي", "الذين", "السابق", "الف", "اللائي", "اللاتي", "اللتان", "اللتيا",
                      "اللتين", "اللذان", "اللذين", "اللواتي", "الماضي", "المقبل", "الوقت", "الى", "اليوم", "اما",
                      "امام", "امس", "ان", "انبرى", "انقلب", "انه", "انها", "او", "اول", "اي", "ايار", "ايام", "ايضا",
                      "ب", "بات", "باسم", "بان", "بخٍ", "برس", "بسبب", "بسّ", "بشكل", "بضع", "بطآن", "بعد", "بعض", "بك",
                      "بكم", "بكما", "بكن", "بل", "بلى", "بما", "بماذا", "بمن", "بن", "بنا", "به", "بها", "بي", "بيد",
                      "بين", "بَسْ", "بَلْهَ", "بِئْسَ", "تانِ", "تانِك", "تبدّل", "تجاه", "تحوّل", "تلقاء", "تلك",
                      "تلكم", "تلكما", "تم", "تينك", "تَيْنِ", "تِه", "تِي", "ثلاثة", "ثم", "ثمّ", "ثمّة", "ثُمَّ",
                      "جعل", "جلل", "جميع", "جير", "حار", "حاشا", "حاليا", "حاي", "حتى", "حرى", "حسب", "حم", "حوالى",
                      "حول", "حيث", "حيثما", "حين", "حيَّ", "حَبَّذَا", "حَتَّى", "حَذارِ", "خلا", "خلال", "دون",
                      "دونك", "ذا", "ذات", "ذاك", "ذانك", "ذانِ", "ذلك", "ذلكم", "ذلكما", "ذلكن", "ذو", "ذوا", "ذواتا",
                      "ذواتي", "ذيت", "ذينك", "ذَيْنِ", "ذِه", "ذِي", "راح", "رجع", "رويدك", "ريث", "رُبَّ", "زيارة",
                      "سبحان", "سرعان", "سنة", "سنوات", "سوف", "سوى", "سَاءَ", "سَاءَمَا", "شبه", "شخصا", "شرع",
                      "شَتَّانَ", "صار", "صباح", "صفر", "صهٍ", "صهْ", "ضد", "ضمن", "طاق", "طالما", "طفق", "طَق", "ظلّ",
                      "عاد", "عام", "عاما", "عامة", "عدا", "عدة", "عدد", "عدم", "عسى", "عشر", "عشرة", "علق", "على",
                      "عليك", "عليه", "عليها", "علًّ", "عن", "عند", "عندما", "عوض", "عين", "عَدَسْ", "عَمَّا", "غدا",
                      "غير", "ـ", "ف", "فان", "فلان", "فو", "فى", "في", "فيم", "فيما", "فيه", "فيها", "قال", "قام",
                      "قبل", "قد", "قطّ", "قلما", "قوة", "كأنّما", "كأين", "كأيّ", "كأيّن", "كاد", "كان", "كانت", "كذا",
                      "كذلك", "كرب", "كل", "كلا", "كلاهما", "كلتا", "كلم", "كليكما", "كليهما", "كلّما", "كلَّا", "كم",
                      "كما", "كي", "كيت", "كيف", "كيفما", "كَأَنَّ", "كِخ", "لئن", "لا", "لات", "لاسيما", "لدن", "لدى",
                      "لعمر", "لقاء", "لك", "لكم", "لكما", "لكن", "لكنَّما", "لكي", "لكيلا", "للامم", "لم", "لما",
                      "لمّا", "لن", "لنا", "له", "لها", "لو", "لوكالة", "لولا", "لوما", "لي", "لَسْتَ", "لَسْتُ",
                      "لَسْتُم", "لَسْتُمَا", "لَسْتُنَّ", "لَسْتِ", "لَسْنَ", "لَعَلَّ", "لَكِنَّ", "لَيْتَ", "لَيْسَ",
                      "لَيْسَا", "لَيْسَتَا", "لَيْسَتْ", "لَيْسُوا", "لَِسْنَا", "ما", "ماانفك", "مابرح", "مادام",
                      "ماذا", "مازال", "مافتئ", "مايو", "متى", "مثل", "مذ", "مساء", "مع", "معاذ", "مقابل", "مكانكم",
                      "مكانكما", "مكانكنّ", "مكانَك", "مليار", "مليون", "مما", "ممن", "من", "منذ", "منها", "مه", "مهما",
                      "مَنْ", "مِن", "نحن", "نحو", "نعم", "نفس", "نفسه", "نهاية", "نَخْ", "نِعِمّا", "نِعْمَ", "ها",
                      "هاؤم", "هاكَ", "هاهنا", "هبّ", "هذا", "هذه", "هكذا", "هل", "هلمَّ", "هلّا", "هم", "هما", "هن",
                      "هنا", "هناك", "هنالك", "هو", "هي", "هيا", "هيت", "هيّا", "هَؤلاء", "هَاتانِ", "هَاتَيْنِ",
                      "هَاتِه", "هَاتِي", "هَجْ", "هَذا", "هَذانِ", "هَذَيْنِ", "هَذِه", "هَذِي", "هَيْهَاتَ", "و",
                      "و6", "وا", "واحد", "واضاف", "واضافت", "واكد", "وان", "واهاً", "واوضح", "وراءَك", "وفي", "وقال",
                      "وقالت", "وقد", "وقف", "وكان", "وكانت", "ولا", "ولم", "ومن", "مَن", "وهو", "وهي", "ويكأنّ",
                      "وَيْ", "وُشْكَانََ", "يكون", "يمكن", "يوم", "ّأيّان"}

    return " ".join([w for w in words if not w in stops and not w in stop_word_comp and len(w) >= 2])



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



def clean_tweet(text):
    text = re.sub('#\d+K\d+', ' ', text)  # years like 2K19
    text = re.sub('http\S+\s*', ' ', text)  # remove URLs
    text = re.sub('RT|cc', ' ', text)  # remove RT and cc
    text = re.sub('@[^\s]+',' ',text)
    text = clean_hashtag(text)
    #text = remove_emoji(text)
    return text


def clean_text(text):
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
    text = remove_stop_words(text)
    ## Remove numbers
    text = re.sub("\d+", " ", text)
    # Stemming
    text = stem(text)
    ## Remove Tashkeel
    text = normalizeArabic(text)
    # text = re.sub('\W+', ' ', text)
    text = re.sub('[A-Za-z]+', ' ', text)
    text = re.sub(r'\\u[A-Za-z0-9\\]+', ' ', text)
    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)

    return text


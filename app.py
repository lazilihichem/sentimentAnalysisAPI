from flask import Flask
import pickle
import numpy as np

from Preprocess import englishPreprocess, arabicPreprocess

app = Flask(__name__)

@app.route("/")
def index():
    app.config['JSON_AS_ASCII'] = False
    return \
        {
            "cleaned-tweet": arabicPreprocess.clean_text("هشام لعزيلي")
        }



@app.route('/arabic-preprocess/<tweet>')
def cleanarab(tweet):
    app.config['JSON_AS_ASCII'] = False

    return \
        {
            "cleaned-tweet": arabicPreprocess.clean_text(tweet)
        }

@app.route('/english-preprocess/<tweet>')
def cleanenglish(tweet):
    app.config['JSON_AS_ASCII'] = False

    return \
        {
            "cleaned-tweet": englishPreprocess.clean_english_text(tweet)
        }



@app.route('/Predict/<tweet>')
def Predict(tweet):
    from tensorflow.python.keras.preprocessing.sequence import pad_sequences
    import tensorflow as tf
    max_lenth = 224
    app.config['JSON_AS_ASCII'] = False
    cleaned_tweet = [englishPreprocess.clean_english_text(tweet)]

    tokenizer_obj = pickle.load(open( "api/tockenizer.p", "rb" )) #load the tokenizer
    treated_sequences = tokenizer_obj.texts_to_sequences(cleaned_tweet)
    paded_sequences = pad_sequences(treated_sequences, maxlen=max_lenth, padding="post")
    model = tf.keras.models.load_model('api/model3.h5')
    y_train_predict = model.predict(paded_sequences)
    y_train_predict = np.argmax(y_train_predict, axis=1 )

    dict = ['negative', 'neutral', 'positive']
    return \
        {
            "sentiment": dict[y_train_predict[0]]

        }

if __name__ == "__main__" :
    app.run(debug=True)
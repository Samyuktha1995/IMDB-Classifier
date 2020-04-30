"""
IMDB Movie review - sentiment analysis using Word2Vec + Bag of Words model for preprocessing the reviews and
Bidirectional LSTM to perform the classification.
Input: labeledTrainData.tsv
Output: submission.csv containing the test data id, review and predictions.
"""
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sb
from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec, Phrases
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences

import time
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

"""
Get word2vec model for the reviews. Convert reviews to trigrams and train word2vec model
"""
class Word2VecModel():
  def __init__(self, all_reviews):
    self.all_reviews = all_reviews
    self.embedding_vector_size = 256
    self.input_length = 150
    self.preprocess = Data_preprocess()

  def getModel(self):
    self.bigrams = Phrases(sentences=self.all_reviews)
    self.trigrams = Phrases(sentences=self.bigrams[self.all_reviews])
    self.word2vec_model = Word2Vec(
        sentences = self.trigrams[self.bigrams[self.all_reviews]],
        size = self.embedding_vector_size,
        min_count=3, window=5, workers=4)
    print("Vocabulary size:", len(self.word2vec_model.wv.vocab))
    return self.word2vec_model

  def convert_ngrams(self, X_train_data):
    X_data = self.trigrams[self.bigrams[X_train_data]]
    X_pad = pad_sequences(
        sequences=self.preprocess.vectorize_data(X_data, vocab=self.word2vec_model.wv.vocab),
        maxlen=self.input_length,
        padding='post')
    return X_pad

"""
Data preprocessing - cleaning the reviews by removing html prefixes, stop words and lemmatizing.
"""
class Data_preprocess():
    def __init__(self):
        var = 0

    # Removing HTML prefixes and non-alphabetical characters
    def clean_review(self, review):
        REPLACE_WITH_SPACE = re.compile(r'[^A-Za-z\s]')
        text = BeautifulSoup(review, "lxml").get_text()
        alphabets = REPLACE_WITH_SPACE.sub(" ", text)
        alphabets = alphabets.lower()
        return alphabets

    # Lemmatize and remove stop-words
    def lemmatize(self, tokens):
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        tokens = list(map(lemmatizer.lemmatize, tokens))
        lemmatized_tokens = list(map(lambda x: lemmatizer.lemmatize(x, "v"), tokens))
        words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
        return words

    # Clean reviews, tokenize and lemmatize
    def preprocess(self, review):
        review = self.clean_review(review)
        tokens = word_tokenize(review)
        lemmas = self.lemmatize(tokens)
        return lemmas

    def vectorize_data(self, data, vocab):
        keys = list(vocab.keys())
        filter_unknown = lambda word: vocab.get(word, None) is not None
        encode = lambda review: list(map(keys.index, filter(filter_unknown, review)))
        vectorized = list(map(encode, data))
        return vectorized

"""
Performs classification of the IMDB review dataset. Loads and preprocesses data and trains NN model. 
"""
class IMDBClassifier():
    def __init__(self):
        self.batch_size = 100
        self.epochs = 20
        self.input_length = 150
        self.preprocess = Data_preprocess()

    def load_dataset(self):
        labeledTrainData = pd.read_csv('Data/labeledTrainData.tsv', usecols=['sentiment', 'review'], sep='\t')
        imdbMaster = pd.read_csv('Data/imdb_master.csv', encoding='iso-8859-1')
        unlabeledTrainData = pd.read_csv('Data/unlabeledTrainData.tsv', error_bad_lines=False, sep='\t')
        self.testData = pd.read_csv('Data/testData.tsv', sep='\t')

        imdbMaster = imdbMaster[imdbMaster.label != 'unsup']
        imdbMaster["sentiment"] = imdbMaster.label.map({"neg": 0, "pos": 1})
        imdbMaster.drop(["type", "label"], axis=1, inplace=True)

        self.all_reviews = []
        for reviews in labeledTrainData['review']:
            self.all_reviews.append(reviews)
        for reviews in imdbMaster['review']:
            self.all_reviews.append(reviews)
        for reviews in unlabeledTrainData['review']:
            self.all_reviews.append(reviews)
        for reviews in self.testData['review']:
            self.all_reviews.append(reviews)

        self.trainData = pd.concat((labeledTrainData, imdbMaster[imdbMaster.sentiment != -1]), axis=0, ignore_index=True)


    def preprocess_data(self):
        self.all_reviews = np.array(list(map(lambda x: self.preprocess.preprocess(x), self.all_reviews)))

    def build_model(self, embedding_matrix, input_length):
        model = Sequential()
        model.add(Embedding(
            input_dim=embedding_matrix.shape[0],
            output_dim=embedding_matrix.shape[1],
            input_length=input_length,
            weights=[embedding_matrix],
            trainable=False))
        model.add(Bidirectional(LSTM(128, recurrent_dropout=0.1)))
        model.add(Dropout(0.25))
        model.add(Dense(64))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='relu'))
        model.summary()
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
        return model

    def plot_curves(self):
        fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        axis1.plot(self.history.history['accuracy'], label='Train', linewidth=3)
        axis1.plot(self.history.history['val_accuracy'], label='Validation', linewidth=3)
        axis1.set_title('Model accuracy', fontsize=16)
        axis1.set_ylabel('accuracy')
        axis1.set_xlabel('epoch')
        axis1.legend(loc='upper left')
        axis2.plot(self.history.history['loss'], label='Train', linewidth=3)
        axis2.plot(self.history.history['val_loss'], label='Validation', linewidth=3)
        axis2.set_title('Model loss', fontsize=16)
        axis2.set_ylabel('loss')
        axis2.set_xlabel('epoch')
        axis2.legend(loc='upper right')
        title = "Results/figure_model2.png"
        plt.savefig(title, dpi=600)
        plt.show()

    def predictions(self):
        test_data = self.testData.review.values
        X_sub = self.w2vModel.convert_ngrams(test_data)
        Y_pred = self.model.predict_classes(X_sub)
        self.testData['sentiment'] = Y_pred
        self.testData.to_csv('Results/submission_model2.csv', index=False, columns=['id', 'sentiment'])

    def run(self):
        self.load_dataset()
        self.preprocess_data()
        self.w2vModel = Word2VecModel(self.all_reviews)
        self.w2v = self.w2vModel.getModel()
        X_train_data = self.all_reviews[:self.trainData.shape[0]]
        Y_train_data = self.trainData.sentiment.values
        X_train_padded = self.w2vModel.convert_ngrams(X_train_data)
        X_train, X_test, y_train, y_test = train_test_split(X_train_padded, Y_train_data, test_size=0.05, shuffle=True,
                                                            random_state=42)
        self.model = self.build_model(self.w2v.wv.vectors, self.input_length)
        self.history = self.model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=self.batch_size, epochs=self.epochs)
        self.plot_curves()
        self.predictions()

if __name__ == "__main__":
    obj = IMDBClassifier()
    obj.run()
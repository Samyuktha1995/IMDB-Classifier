"""
IMDB Movie review - sentiment analysis using Fully Connected NN model.
Input: labeledTrainData.tsv
Output: submission.csv containing the test data id, review and predictions.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import re
import pandas
import numpy
import json
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')

"""
Performs classification of the IMDB review dataset. Loads and preprocesses data and trains NN model. 
The reviews are preprocessed by removing html prefixes, punctuations and stop words.
"""
class ReviewClassifier():
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 174
        self.epochs = 20
        self.batch_size = 512

    def get_data(self):
        self.train_data = pd.read_csv("Data/labeledTrainData.tsv", delimiter='\t')
        self.test_data = pd.read_csv("Data/testData.tsv", delimiter='\t')

    def preprocess(self, review, remove_stopwords=False):
        review_text = BeautifulSoup(review, "html5lib").get_text()
        review_text = re.sub("[^a-zA-Z]", " ", review_text)
        words = review_text.lower().split()
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
        clean_review = ' '.join(words)
        return clean_review

    def get_processed_data(self):
        self.get_data()
        self.clean_train_reviews = []
        self.clean_test_reviews = []
        for review in self.train_data['review']:
            self.clean_train_reviews.append(self.preprocess(review, remove_stopwords=True))
        for review in self.test_data['review']:
            self.clean_test_reviews.append(self.preprocess(review, remove_stopwords=True))

    def tokenize_reviews(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.clean_train_reviews + self.clean_test_reviews)
        text_sequences = tokenizer.texts_to_sequences(self.clean_train_reviews)
        self.train_inputs = pad_sequences(text_sequences, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
        text_sequences = tokenizer.texts_to_sequences(self.clean_test_reviews)
        self.test_inputs = pad_sequences(text_sequences, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')
        self.vocab_size = len(tokenizer.word_index) + 1

    def get_model(self):
        model = Sequential()
        model.add(Embedding(self.vocab_size, 16))
        model.add(GlobalAveragePooling1D())
        model.add(Dense(16, activation=tf.nn.relu))
        model.add(Dense(1, activation=tf.nn.sigmoid))
        print(model.summary())
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def plot_curves(self, history):
        fig, (axis1, axis2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        axis1.plot(history.history['accuracy'], label='Train', linewidth=3)
        axis1.plot(history.history['val_accuracy'], label='Validation', linewidth=3)
        axis1.set_title('Model accuracy', fontsize=16)
        axis1.set_ylabel('accuracy')
        axis1.set_xlabel('epoch')
        axis1.legend(loc='upper left')
        axis2.plot(history.history['loss'], label='Train', linewidth=3)
        axis2.plot(history.history['val_loss'], label='Validation', linewidth=3)
        axis2.set_title('Model loss', fontsize=16)
        axis2.set_ylabel('loss')
        axis2.set_xlabel('epoch')
        axis2.legend(loc='upper right')
        title = "Results/figure_model1.png"
        plt.savefig(title, dpi=600)
        plt.show()

    def get_predictions(self):
        y_pred = np.rint(self.model.predict(self.test_inputs)).astype('int32').squeeze()
        df = pd.DataFrame({'id': self.test_data['id'], 'sentiment': y_pred})
        df.to_csv('Results/submission_model1.csv', index=False)

    def run(self):
        self.get_processed_data()
        self.tokenize_reviews()
        self.model = self.get_model()
        x_train = self.train_inputs
        y_train = np.array(self.train_data['sentiment'])
        history = self.model.fit(x_train, y_train, validation_split=0.2, epochs=self.epochs, batch_size=self.batch_size,
                                 verbose=2)
        self.plot_curves(history)
        self.get_predictions()

if __name__ == "__main__":

    obj = ReviewClassifier()
    obj.run()
from __future__ import print_function, division
import nltk
import os
import random
from collections import Counter
from nltk.corpus import stopwords
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk import word_tokenize, WordNetLemmatizer
stoplist = stopwords.words('english')
def read_data(folder):
    data_list = []
    file_data = os.listdir(folder)
    for data_file in file_data:
        f = open(folder + data_file, 'r',errors='ignore')
        data_list.append(f.read())
    f.close()
    return data_list
#preprocessing the emails and break into tokens 
def preprocess(email):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(str('sentence'))]
#now next is to get the features extracted using bag of words apporach 
def get_features(text, apporach):
    if apporach=='bag_of_words':
        return {word: count for word, count in Counter(preprocess(text)).items() if not word in stoplist}
    else:
        return {word: True for word in preprocess(text) if not word in stoplist}
    #train the model by taking two input features and data samples pratporation for the test data and train data
def train_model(features, data_samples_praportion):
    train_data_size = int(len(features) * data_samples_praportion)
    print(len(features))
    # initialise the training and test sets
    train_data, test_data = features[:train_data_size], features[train_data_size:]
    print ('Training data size = ' + str(len(train_data)) + ' emails')
    print ('Test data size = ' + str(len(test_data)) + ' emails')
    # train the classifier
    print('ram')
    classifier = SklearnClassifier(MultinomialNB())
    classifier.train(train_data)
    return train_data, test_data, classifier

def predict_accuracy(train_data, test_data, classifier):
    # check how the classifier performs on the training and test data
    print ('Predicting Accuracy on the training data set = ' + str(nltk.classify.accuracy(classifier, train_data)))
    print ('Predicting Accuracy on the test data set = ' + str(nltk.classify.accuracy(classifier, test_data)))

if __name__ == '__main__':
 # insering our dataset
    spam = read_data('enron2/spam/')
    ham = read_data('enron2/ham/')
    all_emails = [(email, 'spam') for email in spam]
    all_emails += [(email, 'ham') for email in ham]
    random.shuffle(all_emails)
    print ('data corpus size = ' + str(len(all_emails)) + ' emails')
    # extract the features
    all_features = [(get_features(email, ''), label) for (email, label) in all_emails]
    print ('Caculate ' + str(len(all_features)) + ' feature sets')
    # train the model 
    train_data, test_data, classifier = train_model(all_features, 0.8)

    # calculate the accuracy 
    predict_accuracy(train_data, test_data, classifier)

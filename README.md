
# Email-Spam-Detection-
Here i have used python 2.7 to code the model however if you re using python 3.5 or higher must use 'str' function  instead of 'unicode' 
i have make both training and testing in a single program to run this model simply run the email_spam_detection.py 
make sure u have installed all the dependacies mentioned in the code 
code is flexible such that it will download stopwords and wordnet corpus if it is not already availabe in system
the reason i have chosen the Naivebayes classifier becausem The Naive Bayes classifier has been shown to perform surprisingly well with very small amounts of training data that most other classifiers i have used nltk toolkit to tokenize and lemmatize the word i have  used bag of words apporach because am intrested in finding current  word when given previous and nextwords for text classifaiction 
in the last i have trained the model and predict the accuracy it will work well on sparse dataset where emails in spam and ham folder are uneven i have got 94.6% test accuracy on the enron dataset 
still i can improve it in the next commit 

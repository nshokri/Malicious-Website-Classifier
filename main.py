# Author: Nick Shokri
# Date: 12/14/2020
# Class: CSS 486

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree

# You only need these imports if you want to try the live demo, otherwise just leave them commented
'''
from ipwhois import IPWhois # Note: you need to uncomment the block of code at the bottom of the file as well, if you want to use this
import requests
import socket
'''

# Machine learning object that classifies whether websites are malicious or benign
class MaliciousWebsiteClassifier:
    #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    # Uses the K-Nearest Neighbor algorithm with k = 5 and returns the f1-score of the model
    def kNearestNeighbor(self, X_train, X_test, y_train, y_test):
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X_train, y_train)

        expected = y_test
        predicted = neigh.predict(X_test)

        metric = metrics.f1_score(expected, predicted)

        return [neigh, metric]

    # https://scikit-learn.org/stable/modules/naive_bayes.html
    # Uses the Bayesian Classifier algorithm with k = 5 and returns the f1-score of the model
    def bayesianClassification(self, X_train, X_test, y_train, y_test):
        # Use the Naive Baise algorithm
        gnb = GaussianNB()

        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        expected = y_test

        metric = metrics.f1_score(expected, y_pred)

        return [gnb, metric]


    # https://scikit-learn.org/stable/modules/tree.html
    # Uses the Decision Tree algorithm with k = 5 and returns the f1-score of the model
    def decisionTree(self, X_train, X_test, y_train, y_test):
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, y_train)

        predict = clf.predict(X_test)
        expected = y_test

        metric = metrics.f1_score(expected, predict)

        return [clf, metric]

    # Encodes categorical variables into numeric
    def encodeData(self, data):

        for feature in data:
            data[feature] = LabelEncoder().fit_transform(data[feature])

        return data
    # Predicts the class of a website using the best model found from the data
    def predict(self, data):
        data = self.encodeData(data)
        data = preprocessing.scale(data)
        return self.model.predict(data)

    # Outputs the predict_proba of the given data using the model
    def predictProb(self, data):
        data = self.encodeData(data)
        return self.model.predict_proba(data)

    def __init__(self, datasetName):
        '''Pre-processing'''
        data = pd.read_csv(datasetName, sep=",")

        # Drop URL because it is hidden/non-accessible, drop server because we can't parse it with the encoder
        data = data.drop(["SERVER"], 1)
        data = data.drop(["URL"], 1)

        data = self.encodeData(data)

        # Type is equal to 1 for malicious and 0 for benign
        predict = "Type"

        # X = all features except what we want to guess, y = what we want to guess
        X = np.array(data.drop([predict], 1))
        y = np.array(data[predict])

        # Remove features with low variance: https://scikit-learn.org/stable/modules/feature_selection.html
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        sel.fit_transform(X)

        # Feature select the most impactful features: https://scikit-learn.org/stable/modules/feature_selection.html
        X = SelectKBest(chi2, k=16).fit_transform(X, y)


        # Normalize the data
        X = preprocessing.scale(X)

        # Split 50% of the data for training and the rest for validation
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=5)

        # https://www.youtube.com/watch?v=gJo0uNL-5Qw
        # Cross validation using: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        kf = KFold(n_splits=10)
        kf.get_n_splits(X)
        scores = []
        models = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            kNear = self.kNearestNeighbor(X_train, X_test, y_train, y_test)
            NB = self.bayesianClassification(X_train, X_test, y_train, y_test)
            tree = self.decisionTree(X_train, X_test, y_train, y_test)
            models.append(kNear[0])
            models.append(NB[0])
            models.append(tree[0])
            scores.append(kNear[1])
            scores.append(NB[1])
            scores.append(tree[1])

        knear_score = []
        bayes_score = []
        tree_score = []

        # Print results
        for i in range (len(scores)):
            if (i % 3 == 0):
                knear_score.append(scores[i])
            elif (i % 3 == 1):
                bayes_score.append(scores[i])
            else:
                tree_score.append(scores[i])

        print('K-Nearest Neighbor f1-scores: ' + str(knear_score))
        print('Naive Bayes f1-scores: ' + str(bayes_score))
        print('Decision Tree f1-scores: ' + str(tree_score))

        # Pick the best results
        bestScore = -1
        bestIndex = -1
        for i in range(len(scores)):
            if (scores[i] > bestScore):
                bestScore = scores[i]
                bestIndex = i

        # Pick the model with the best f1-score that we got from cross validation
        self.model = models[bestIndex]

        if (bestIndex % 3 == 0):
            print('Best Model: K-NearestNeighbor')
        elif (bestIndex % 3 == 1):
            print('Best Model: Naive Bayes')
        else:
            print('Best Model: decision trees')


# NOTE: ONLY UNCOMMENT IF YOU WANT TO TRY THE LIVE DEMO AND YOU HAVE UNCOMMENTED THE IMPORT STATEMENTS
# The demo requires an internet connection
    '''
# The website URL format should be this exactly: 'https://www.kaggle.com/' MUST HAVE BE A '.com' WEBSITE
website_url = input('Enter a website URL: ')

url_length = len(website_url)

num_special_char = 0
for n in website_url:
    if (n == '/' or n == '!' or n == '@' or n == '#' or n == '$' or n == '%' or n == '^' or
    n == '&' or n == '*' or n == '(' or n == ')' or n == '.' or n == '='):
        num_special_char += 1

try:
    response = requests.get(website_url, stream=True)
    header = response.headers
except:
    pass

content_length = 'NA'
charset = 'None'
ip = None

try:
    charset = header['Content-Type'].split(';')[1].split('=')[1]
except:
    pass

try:
    #content_length = header['Content-length']
    content_length = len(header)
except:
    pass

try:
    temp = website_url[website_url.find('//') + 2 : website_url.find('.com') + 4]
    ip = socket.gethostbyname(website_url[website_url.find('//') + 2 : website_url.find('.com') + 4])
except:
    pass

try:
    ip = socket.gethostbyname_ex(website_url)
except:
    pass

country = 'None'
state = 'None'
registration_date = 'None'
update_date = 'None'

look_up_location = None
look_up_extra = None
try:
    who_is = IPWhois(ip)
    look_up_location = who_is.lookup_whois()
    look_up_extra = who_is.lookup_rdap(asn_methods=['whois'])

    country = look_up_location['nets'][0]['country']
    state = look_up_location['nets'][0]['state']
    update_date = look_up_location['nets'][0]['updated']

    for n in look_up_extra['network']['events']:
        if (n['action'] == 'registration'):
            registration_date = n['timestamp']

except:
    pass
# Make sure you format registration date correctly

day = ((registration_date[:10])[-2:])
month = ((registration_date[:9])[5:7])
year = ((registration_date[:9])[0:4])
time = registration_date[-5:]

registration_date = day + '/' + month + '/' + year + ' ' + time

# Unfortunately I can't find a way to get these features just from the URL alone, but if you want to
# test it out on a server with known properties, feel free to change these values to get the most
# accurate result from the model
tcp_conversion_rate = '0'
remote_distance = '0'
remote_ip = '0'
app_bytes = '0'
source_app_packets = '0'
remove_app_packets = '0'
dns_time = '0'


model = MaliciousWebsiteClassifier('dataset.csv')

data = [[website_url, url_length, num_special_char, charset, content_length, country, state, registration_date, update_date, tcp_conversion_rate, remote_distance, remote_ip, app_bytes, source_app_packets, remove_app_packets, dns_time]]

data = pd.DataFrame(data)

print('Prediction: ' + str(model.predict(data)))
'''

# Comment out this line if you want to try the live demo
website_classifier = MaliciousWebsiteClassifier('dataset.csv')

'''
Description:
Last update: 19th April 2020
- From p.9 of YouTube tutorial
- Up to p.10 of YouTube tutorial
'''
from collections import Counter
import numpy as np
import pandas as pd
import pickle
from sklearn import svm,model_selection, neighbors
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_combined.csv',index_col=0)
    tickers = df.columns.values
    df.fillna(0,inplace=True)

    for i in range(1,hm_days+1):
        df['{}_{}d'.format(ticker,i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]

    df.fillna(0,inplace=True)
    return tickers,df

#process_data_for_labels('XOM')

def buy_sell_hold(*args):  #can pass anything to the arguments
    cols = [c for c in args]
    requirement = 0.04  #2 percent change in stock price
    for col in cols:
        if col > requirement:
            return 1
        if col < requirement:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers,df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold,df['{}_1d'.format(ticker)],
                                                            df['{}_2d'.format(ticker)],
                                                            df['{}_3d'.format(ticker)],
                                                            df['{}_4d'.format(ticker)], 
                                                            df['{}_5d'.format(ticker)], 
                                                            df['{}_6d'.format(ticker)],
                                                            df['{}_7d'.format(ticker)]))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals)) 

    df.fillna(0, inplace = True)
    df = df.replace([np.inf, - np.inf], np.nan) # replace the infinitie values with nan
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, - np.inf], 0)
    df_vals.fillna(0,inplace=True)

    #Set Features and labels
    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X,y,df

def do_ml(ticker):
    X,y,df = extract_featuresets(ticker)

    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.25)

    #clf = neighbors.KNeighborsClassifier()
    clf = VotingClassifier([('lsvc',svm.LinearSVC()),
                            ('knn',neighbors.KNeighborsClassifier()),
                            ('rfor',RandomForestClassifier())])
    clf.fit(X_train,y_train)
    confidence = clf.score(X_test,y_test)
    print('Accuracy',confidence)
    predictions = clf.predict(X_test)
    print('Predicted Spread:', Counter(predictions))

    return confidence

do_ml('MSFT')


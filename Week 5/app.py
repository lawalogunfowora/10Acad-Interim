from flask import Flask, render_template
from flask_cors import CORS
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from scripts import multipleTransformer
from fbprophet import Prophet
import pandas as pd
import numpy as np
import logging
import pickle
import datetime
port = 8919

form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
logger = logging.getLogger()


consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(form)
logger.addHandler(consoleHandler)

logger.setLevel(logging.DEBUG)

app = Flask(__name__)
CORS(app)

# set this by default to allow cross domain calls to work
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/', methods=['POST', 'GET'])
def index():

    df = pd.read_csv('testStore.csv')
    dfLine = pd.read_csv('dateForecast.csv')

    return render_template('index.html', labels=df['Id'], values=df['Sales'],
                           linelabels=dfLine['ds'], linevalues=dfLine['yhat'])

@app.route('/model.html', methods=['POST', 'GET'])
def runmodels():
    # load the data
    dfTrain = pd.read_csv('train.csv', low_memory=False)
    dfTest = pd.read_csv('test.csv', low_memory=False)
    dfStore = pd.read_csv("store.csv", low_memory=False)

    # dropping the zero sales and closed stores
    dfTrain = dfTrain[(dfTrain.Open != 0) & (dfTrain.Sales != 0)]

    sales, holidays = prophetData(dfTrain)

    # filling the NaN values in CompetitionDistance col
    dfStore.CompetitionDistance.fillna(dfStore.CompetitionDistance.median(), inplace=True)

    # replace all the other NaN values with zeros
    dfStore.fillna(0, inplace=True)

    # fill the missing values
    dfTest.fillna(1, inplace=True)

    # merge train and test dataset with store data
    dfTrainStore = merge(dfTrain, dfStore)
    dfTestStore = merge(dfTest, dfStore)

    # Set the target column
    Y = dfTrainStore['Sales']
    Id = dfTestStore['Id']

    # remove dataset specific columns
    dfTrainStore = dfTrainStore.drop(['Customers', 'Sales'], axis=1)
    dfTestStore = dfTestStore.drop(['Id'], axis=1)

    # split the data into a training set and a validation set
    xTrain, xTrainTest, yTrain, yTrainTest = train_test_split(dfTrainStore, Y, test_size=0.20, random_state=42)

    pipe = Pipeline(steps=[
        ('multipleTrans', multipleTransformer()),
        ('randomForest', RandomForestRegressor(n_estimators=128,
                                               criterion='mse',
                                               max_depth=20,
                                               min_samples_split=10,
                                               min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0,
                                               max_features='auto',
                                               max_leaf_nodes=None,
                                               min_impurity_decrease=0.0,
                                               min_impurity_split=None,
                                               bootstrap=True,
                                               oob_score=False,
                                               n_jobs=4,
                                               random_state=35,
                                               verbose=0,
                                               warm_start=False))
    ])

    regModel = TransformedTargetRegressor(regressor=pipe, func=targetTransform,
                                          inverse_func=reverseTargetTransform)

    # training the Regression Model
    regModel.fit(xTrain, yTrain)

    # Regression Model prediction
    yPred = regModel.predict(xTrainTest)

    # predict on the testStore set
    predictions = regModel.predict(dfTestStore)

    # turn the predictions into a dataframe
    dfPreds = pd.DataFrame({'Id': Id,
                            'Sales': predictions})

    # training the prophet Model
    pModel = Prophet(interval_width=0.5, holidays=holidays)
    pModel.fit(sales)

    # dataframe that extends into future 6 weeks
    future_dates = pModel.make_future_dataframe(periods=6 * 7)

    # prophet model predictions
    forecast = pModel.predict(future_dates)

    # rename prediction columns and isolate the predictions
    fc = forecast[['ds', 'yhat']].rename(columns={'Date': 'ds', 'Forecast': 'yhat'})

    # get the current time and turn it into a string
    now = datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S-%f')[:-3]

    # Save the model
    filenameReg = 'regModel-' + now + '.pkl'
    filenamePro = 'pModel-' + now + '.pkl'
    pickle.dump(regModel, open(filenameReg, 'wb'))
    pickle.dump(pModel, open(filenamePro, 'wb'))

    return render_template('model.html', labels=dfPreds['Id'], values=dfPreds['Sales'],
                           linelabels=fc['ds'], linevalues=fc['yhat'])

# merge the Store data and the train data
def merge(df, store):
    logger.info("\n merging datasets \n")
    dfJoined = pd.merge(df, store, how='inner', on='Store')
    return dfJoined

# transform the target column
def targetTransform(target):
    logger.info('\n transforming the target Column \n')
    target = np.log(target)
    return target

# reverse target transform
def reverseTargetTransform(target):
    logger.info('\n reverse transforming the target col \n')
    target = np.exp(target)
    return target

# error calculation
def rmspe(y, yhat):
    rmspe = np.sqrt(mean_absolute_error(y, yhat))
    return rmspe

def prophetData(df):
    logger.info('\n Making the prophet data')
    # sales for the store number 1 (StoreType C)
    sales = df[df.Store == 1].loc[:, ['Date', 'Sales']]

    # reverse to the order: from 2013 to 2015
    sales = sales.sort_index(ascending=False)

    # to datetime64
    sales['Date'] = pd.DatetimeIndex(sales['Date'])

    # from the prophet documentation every variables should have specific names
    sales = sales.rename(columns={'Date': 'ds', 'Sales': 'y'})

    # create a holidays dataframe
    state_dates = df[(df.StateHoliday == 'a') | (df.StateHoliday == 'b')
                     & (df.StateHoliday == 'c')].loc[:, 'Date'].values
    school_dates = df[df.SchoolHoliday == 1].loc[:, 'Date'].values

    state = pd.DataFrame({'holiday': 'state_holiday', 'ds': pd.to_datetime(state_dates)})
    school = pd.DataFrame({'holiday': 'school_holiday', 'ds': pd.to_datetime(school_dates)})

    holidays = pd.concat((state, school))

    return sales, holidays


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)

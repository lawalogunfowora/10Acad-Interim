import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin

form = logging.Formatter("%(asctime)s : %(levelname)-5.5s : %(message)s")
logger = logging.getLogger()


consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(form)
logger.addHandler(consoleHandler)

logger.setLevel(logging.DEBUG)


class multipleTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        logger.info('\n >>>>> init() called. \n')

    def fit(self, X, y=None):
        logger.info('\n >>>>> fit() called. \n')
        return self

    # add aditional columns to the dataset.
    def dataTuning(self, df):
        logger.info('\n creating the data Year, Month, Week of Year columns \n')
        df['Date'] = pd.to_datetime(df.Date)
        df.set_index('Date', inplace=True)
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['WeekOfYear'] = df.index.weekofyear

        return df

    # Create dummy variables instead of binary encoding to avoid uneven weight distribution in the categorical cols
    def createDummies(self, df):
        logger.info('\n Creating Dummie columns ')
        df = pd.get_dummies(df, columns=["Assortment", "StoreType", "PromoInterval"],
                            prefix=["is_Assortment", "is_StoreType", "is_PromoInteval"])
        return df

    # creates a new column that combines the CompetitionOpenSinceMonth and CompetitionOpenSinceYear columns
    def compeSince(self, df):
        logger.info('\n running CompeSince \n')
        df['CompetitionOpenSince'] = np.where((df['CompetitionOpenSinceMonth'] == 0) &
                                              (df['CompetitionOpenSinceYear'] == 0), 0,
                                              (df.Month - df.CompetitionOpenSinceMonth) +
                                              (12 * (df.Year - df.CompetitionOpenSinceYear)))
        return df

    # This function sets various columns into categorical
    def setCat(self, df):
        logger.info('\n running Set to Category \n')
        df['StateHoliday'] = df['StateHoliday'].astype('category')
        df['Assortment'] = df['Assortment'].astype('category')
        df['StoreType'] = df['StoreType'].astype('category')
        df['PromoInterval'] = df['PromoInterval'].astype('category')

        return df

    # change stateHoliday to 0 1 values
    def stateHol(self, df):
        logger.info('\n Change state Holiday to 0 and 1 \n')
        df["is_holiday_state"] = df['StateHoliday'].map({"0": 0, "a": 1, "b": 1, "c": 1})
        return df

    # this function drops columns
    def dropCol(self, df):
        logger.info('\n dropping unecessary columns \n')
        df = df.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'StateHoliday'], axis=1)
        return df

    def transform(self, X, y=None):
        logger.info("\n the transform function has been called \n")

        X = self.dataTuning(X)
        X = self.compeSince(X)
        X = self.setCat(X)
        X = self.stateHol(X)
        X = self.createDummies(X)
        X = self.dropCol(X)

        return X

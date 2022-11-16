import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

# acquire
from env import get_db_url
from pydataset import data
import seaborn as sns

# turn off pink warning boxes
import warnings
warnings.filterwarnings("ignore")

def wrangle_zillow():
    '''
    This function reads the zillow data from the Codeup db into a df.
    '''
    sql_query = """
                SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                FROM properties_2017
                WHERE propertylandusetypeid = 261
                """
    
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_db_url('zillow'))
    # replace any whitespace with null values
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    # drop out any null values:
    df = df.dropna()
    # cast everything as an integer:
    df = df.astype(int)
    
    #@@@@@@@@ added later, may need to be deleted
    df = df.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 'calculatedfinishedsquarefeet': 'square_feet', 'taxvaluedollarcnt': 'property_value'})
    df.drop(df[df.square_feet > 70000].index, inplace=True)
    df.drop(df[df.property_value > 50000000].index, inplace=True)
    df.drop(df[df.taxamount > 400000].index, inplace=True)
    df['baseline'] = df.square_feet.mean()
    
    lm = LinearRegression()

    # fit the model to trainig data
    lm.fit(df[['property_value']], df.square_feet)

    # make prediction
    # lm.predict will output a numpy array of values,
    # we will put those values into a series in df
    #df['yhat'] = lm.predict(df[['property_value']])
    
    #df['baseline_residual'] = df.square_feet - df.baseline
    #df['residual'] = df.square_feet - df.yhat
    
    #df['baseline_residual_2'] = df.baseline_residual**2
    #df['residual_2'] = df.residual**2
    
    return df

def split_zillow(df):
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=123)
    return train, validate, test


######


def get_fips(train):
    sns.relplot(x='property_value', y='fips', data=train)
    plt.title('Fips vs. Property Value')
    plt.show
    
def get_chi_fips(train):
    observed = pd.crosstab(train.fips, train.property_value)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    

    
def get_square_feet(train):
    sns.relplot(x='property_value', y='square_feet', data=train)
    plt.title('Sqaure Feet vs. Property Value')
    plt.show
    

    
    
    
    
def get_year_built(train):
    sns.relplot(x='property_value', y='yearbuilt', data=train)
    plt.title('Year Built vs. Property Value')
    plt.show
    
def get_chi_year_built(train):
    observed = pd.crosstab(train.yearbuilt, train.property_value)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    
    
    
def get_bathrooms(train):
    sns.boxplot(data=train, y='property_value', x='bathrooms')
    plt.title('Bathrooms vs. Property Value')
    plt.show
    
def get_chi_bathrooms(train):
    observed = pd.crosstab(train.bathrooms, train.property_value)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    
    
    
def get_bedrooms(train):
    sns.boxplot(data=train, y='property_value', x='bedrooms')
    plt.title('Bedrooms vs. Property Value')
    plt.show
    
def get_chi_bedrooms(train):
    observed = pd.crosstab(train.bedrooms, train.property_value)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    
    
    
    
# prep data for modeling
def model_prep(train, validate, test):
    '''Prepare train, validate, and test data for modeling'''
    
    # drop unused columns
    keep_cols = ['property_value',
                 'fips',
                 'yearbuilt',
                 'bathrooms',
                 'bedrooms'
                ]
    
    train = train[keep_cols]
    validate = validate[keep_cols]
    test = test[keep_cols]
    
    # Split data into predicting variables (X) and target variable (y) and reset the index for each dataframe
    train_X = train.drop(columns='property_value').reset_index(drop=True)
    train_y = train[['property_value']].reset_index(drop=True)
    
    validate_X = validate.drop(columns='property_value').reset_index(drop=True)
    validate_y = validate[['property_value']].reset_index(drop=True)
    
    test_X = test.drop(columns='property_value').reset_index(drop=True)
    test_y = test[['property_value']].reset_index(drop=True)
    
    return train_X, validate_X, test_X, train_y, validate_y, test_y




def scale_data(train_X, validate_X, test_X):
    # Scale the data
    scaler = sklearn.preprocessing.MinMaxScaler()

    # Fit the scaler
    scaler.fit(train_X)

    # Use the scaler to transform train, validate, test
    X_train_scaled = scaler.transform(train_X)
    X_validate_scaled = scaler.transform(validate_X)
    X_test_scaled = scaler.transform(test_X)


    # Turn everything into a dataframe
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_X.columns)
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns=train_X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=train_X.columns)
    return X_train_scaled, X_validate_scaled, X_test_scaled




def get_mean(train_y, validate_y):
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values.
    y_train = pd.DataFrame(train_y)
    y_validate = pd.DataFrame(validate_y)
    
    # Predict property_value_pred_mean
    property_value_pred_mean = train_y.property_value.mean()
    train_y['property_value_pred_mean'] = property_value_pred_mean
    validate_y['property_value_pred_mean'] = property_value_pred_mean
    
    # compute property_value_pred_median
    property_value_pred_median = train_y.property_value.median()
    train_y['property_value_pred_median'] = property_value_pred_median
    validate_y['property_value_pred_median'] = property_value_pred_median
    
    # RMSE of property_value_pred_mean
    rmse_train = mean_squared_error(y_train.property_value,
                                y_train.property_value_pred_mean) ** .5
    rmse_validate = mean_squared_error(y_validate.property_value, y_validate.property_value_pred_mean) ** (1/2)
    
    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    # RMSE of property_value_pred_median
    rmse_train = mean_squared_error(y_train.property_value, y_train.property_value_pred_median) ** .5
    rmse_validate = mean_squared_error(y_validate.property_value, y_validate.property_value_pred_median) ** .5
    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
      "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))
    
    
    
    
def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                'RMSE_validate': mean_squared_error(
                    y,
                    y_pred) ** .5,
                'r^2_validate': explained_variance_score(
                    y,
                    y_pred)
            }, ignore_index=True)
    
    
    
    
def linear_regression(train_X, train_y, validate_X, validate_y):
    lm = LinearRegression(normalize=True)
    lm.fit(train_X, train_y.property_value)
    train_y['property_value_pred_lm'] = lm.predict(train_X)
    
    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.property_value, train_y.property_value_pred_lm) ** (1/2)

    # predict validate
    validate_y['property_value_pred_lm'] = lm.predict(validate_X)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.property_value, validate_y.property_value_pred_lm) ** (1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate




def lassolars(train_X, train_y, validate_X, validate_y):
    # create the model object
    lars = LassoLars(alpha=1)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series!
    lars.fit(train_X, train_y.property_value)

    # predict train
    train_y['property_value_pred_lars'] = lars.predict(train_X)

    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.property_value, train_y.property_value_pred_lars) ** (1/2)

    # predict validate
    validate_y['property_value_pred_lars'] = lars.predict(validate_X)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.property_value, validate_y.property_value_pred_lars) ** (1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate




def tweedie(train_X, train_y, validate_X, validate_y):
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)


    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(train_X, train_y.property_value)

    # predict train
    train_y['property_value_pred_glm'] = glm.predict(train_X)

    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.property_value, train_y.property_value_pred_glm) ** (1/2)

    # predict validate
    validate_y['property_value_pred_glm'] = glm.predict(validate_X)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.property_value, validate_y.property_value_pred_glm) ** (1/2)

    print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate




def linear(train_X, train_y, validate_X, validate_y, test_X):
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(train_X)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(validate_X)
    X_test_degree2 =  pf.transform(test_X)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, train_y.property_value)

    # predict train
    train_y['property_value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(train_y.property_value, train_y.property_value_pred_lm2) ** (1/2)

    # predict validate
    validate_y['property_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(validate_y.property_value, validate_y.property_value_pred_lm2) ** 0.5

    print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
      "\nValidation/Out-of-Sample: ", rmse_validate)
    return rmse_train, rmse_validate
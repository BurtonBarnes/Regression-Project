import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy import stats

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
    
    
    
    

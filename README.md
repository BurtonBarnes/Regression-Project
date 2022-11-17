# Single Family Properties
# Project Description

Zillow is a real estate company and as such we must get better predictions for the value of properties we will be listing. We will be comparing the home value with several features to understand what could be used to predict a homes price in a negative or positive manner.

# Project Goal

* Discover drivers of property value from zillow database.
* Use drivers to develop machine learning model to classify property value drivers as positive or negative.
* This information could be used to further our understanding of which elements contribute to or detract from a houses property value.

# Initial Thoughts

My initial hypothesis is that what will detract from property value are negative features and what will make the property value higher is positive features. I believe that the higher the square feet, number or bedrooms, and number of bathrooms the higher the property value. With fips I am unsure since the description of the fips area is lacking in descriptors for me to base it on. With year that the house was built I believe that the newer the house the more expensive it will be.


# The Plan

* Aquire data

* Explore data in search of drivers of property value
    * Answer the following initial questions
        * Does fips affect property value?
        * Does bathrooms affect property value?
        * Does bedrooms affect property value?
        * Does square feet affect property value?
        * Does year built affect property value?
        
* Develop a Model to predict the value of a property
    * Use drivers identified in explore to build predictibve models of different types
    * Evaluate models on train and validate data
    * Select the best model on test data
    
* Draw Conclusions

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|bedrooms|  Number of bedrooms in home |
|bathrooms|  Number of bathrooms in home including fractional bathrooms |
|fips|  Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details |
|taxamount|  The total property tax assessed for that assessment year |
|year_built|  The Year the principal residence was built |
|square_feet|Calculated total finished living area of the home |
|property_value|  The total tax assessed value of the parcel |

# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from SQL
3) Put the data in the file containing the cloned repo.
4) Run notebook.

# Takeaways and Conclusions
* Those with a higher number of bathrooms have a higher property value
* Those with a higher number of bedrooms have a higher property value
* Those with a lower fips have a higher property value
* Those with a higher year built house have a higher property value
* Those with a higher square feet have a higher property value
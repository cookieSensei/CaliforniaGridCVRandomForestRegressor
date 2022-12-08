import joblib

from PIL import Image

import streamlit as st
import shap

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import matplotlib.pyplot as plt
#####################



#########################

housing = pd.read_csv("housing.csv")
print(housing.columns)

housing["income_cat"] = pd.cut(housing["median_income"],
                              bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                              labels=list(range(1,6)))
#########################

housing = housing.drop("median_house_value", axis=1)

housing_num = housing.drop("ocean_proximity", axis=1)

# Let's get the column index
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices
# print(rooms_ix, bedrooms_ix, population_ix, households_ix)

############
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self
        # nothing else to do,
        # Just following the boss scikit-learn order's here
        # To be consistent in the api call

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
 #########

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])



num_attribs = list(housing_num)

cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
# print(pd.DataFrame(housing_prepared))

print(housing)

grid_search = joblib.load("RandomForestRegressorGridCV")


final_predictions = grid_search.predict(housing_prepared[0].reshape(1,-1))

print(final_predictions)




################
####################
st.header("California House Price Predictions")
st.write("""
California House Price Prediction App
This app predicts the **California House Price**!
""")

st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    longitude = st.sidebar.slider("longitude:",
                                  float(housing.longitude.min()),
                                  float(housing.longitude.max()),
                                  float(housing.longitude.mean()))
    latitude = st.sidebar.slider("latitude:",
                                  float(housing.latitude.min()),
                                  float(housing.latitude.max()),
                                  float(housing.latitude.mean()))
    housing_median_age = st.sidebar.slider("Median age of a house within a town:",
                                 float(housing.housing_median_age.min()),
                                 float(housing.housing_median_age.max()),
                                 float(housing.housing_median_age.mean()))
    total_rooms = st.sidebar.slider("Total number of rooms within a town:",
                                 float(housing.total_rooms.min()),
                                 float(housing.total_rooms.max()),
                                 float(housing.total_rooms.mean()))
    total_bedrooms = st.sidebar.slider("Total number of bedrooms within a town:",
                                 float(housing.total_bedrooms.min()),
                                 float(housing.total_bedrooms.max()),
                                 float(housing.total_bedrooms.mean()))
    population = st.sidebar.slider("Total number of people residing within a town:",
                                 float(housing.population.min()),
                                 float(housing.population.max()),
                                 float(housing.population.mean()))
    households = st.sidebar.slider("Total number of household within a town:",
                                 float(housing.households.min()),
                                 float(housing.households.max()),
                                 float(housing.households.mean()))
    median_income = st.sidebar.slider("Median income for households within a block of houses",
                                 float(housing.median_income.min()),
                                 float(housing.median_income.max()),
                                 float(housing.median_income.mean()))
    ocean_proximity = st.sidebar.selectbox("Location of the house w.r.t ocean/sea:",
                                           ( 'ISLAND', 'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN'))

    data = {'longitude':longitude,
            'latitude':latitude,
            'housing_median_age':housing_median_age,
            'total_rooms':total_rooms,
            'total_bedrooms':total_bedrooms,
            'population':population,
            'households':households,
            'median_income':median_income,
            'ocean_proximity':ocean_proximity}
    return pd.DataFrame(data, index=[0])

housing = user_input_features()
housing["income_cat"] = pd.cut(housing["median_income"],
                              bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                              labels=list(range(1,6)))

housing_prepared = full_pipeline.transform(housing)
final_predictions = grid_search.predict(housing_prepared[0].reshape(1,-1))

print(final_predictions)
#########################


# Print specified input parameters
st.header('Specified Input parameters')
st.write(housing)
st.write('---')


st.header('Your dream house would cost approximately:')
st.write(final_predictions)
st.write('---')

##############
st.header("Population Density vs House Value")
image = Image.open("California Town Density.jpg")
st.image(image)
# st.image(image, caption='Population Density vs House Value')


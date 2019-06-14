#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pylab as plt

# Read csv
# Add headers
# Prerocessing - Drop coloum with missing value
#              - Replace missing values with mean, frquency
#              - Normalize data
#              -Convert data to correct type
#              - Data standaerdization


#read csv
df= pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", header= None)
#df.tail()

#add headers
headers= ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

df.columns = headers


"""------------------------------MissinG Values--------------------------------- """
#delete rows with  missing values
df['price'].replace("?", np.nan, inplace=True)
df.dropna(subset=['price'], axis=0, inplace=True)

# reset index
df.reset_index(drop=True, inplace=True)

#peak-rpm pre-processing
df["peak-rpm"].replace("?", np.nan, inplace=True)
mean=df["peak-rpm"].astype("float").mean(axis=0)
df["peak-rpm"].replace(np.nan, mean, inplace=True)

#missing values for bore
df["bore"].replace("?", np.nan, inplace=True)
mean= df["bore"].astype("float").mean(axis=0)
df["bore"].replace(np.nan, mean, inplace=True)

#missing values for horse power
df["horsepower"].replace("?", np.nan, inplace=True)
mean= df["horsepower"].astype("float").mean(axis=0)
df["horsepower"].replace(np.nan, mean, inplace=True)

#replacing values for stroke
df["stroke"].replace("?", np.nan, inplace=True)
mean = df["stroke"].astype(float).mean(axis=0)
df["stroke"].replace(np.nan, mean, inplace=True)

#replace the missing question marks with NaN
df['normalized-losses'].replace("?", np.nan, inplace=True)
mean =df['normalized-losses'].astype(float).mean(axis=0)
df['normalized-losses'].replace(np.nan, mean, inplace=True)

#replacing value with frequency
df["num-of-doors"].replace("?", np.nan, inplace=True)
mode= df["num-of-doors"].mode()
df["num-of-doors"].replace(np.nan, "four", inplace=True)

"""__________________________________- Normalization-_______________________________________"""
##min max normalization 

#width
df["width"]= (df["width"]-df["width"].min())/(df["width"].max()- df["width"].min())

#length
df["length"]= (df["length"]-df["length"].min())/(df["length"].max()- df["length"].min())

#height
df["height"]= (df["height"]-df["height"].min())/(df["height"].max()- df["height"].min())


#Setting coloums to the correct data types
df[["normalized-losses"]]= df[["normalized-losses"]].astype("int")
df[["bore", "stroke", "price", "peak-rpm"]] = df[["bore", "stroke", "price", "peak-rpm"]].astype("float")


#####data standerdization 

df["highway-mpg"] = 235/df["highway-mpg"]
df.rename(columns={"highway-mpg":"highway-L/100km"}, inplace=True)

df["city-mpg"] = (235/df["city-mpg"])
df.rename(columns ={"city-mpg": "city-l/1000km"}, inplace=True)

#round decimals
df = df.round(2)


########creating dummy for fuel type
df.columns
dummy = pd.get_dummies(df['fuel-type'])

#merging dummy with dataframe and removing coloum fueel type
df = pd.concat([df, dummy],axis = 1)
df.drop('fuel-type', axis=1, inplace=True)


#pre-processed Csv
df.to_csv('processed.csv')






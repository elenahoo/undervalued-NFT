###################
## Load packages ##
###################

import io
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn.model_selection import KFold
from graphviz import Source
from sklearn import tree
from sklearn import datasets, ensemble, preprocessing

## Functions
def plot_confusion_matrix(cm, classes=None, perc = True, title = 'my title'):
    """Plots a confusion matrix."""
    if classes is not None:
        ax = sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, cmap='Greens')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)
        for t in ax.texts: t.set_text(t.get_text()+"%")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)


    else:
        ax = sns.heatmap(cm, vmin=0., vmax=1., cmap = 'Greens')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom+0.5, top-0.5)
        for t in ax.texts: t.set_text(t.get_text()+"%")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(title)



def plot_feature_importance(model, feature_name = None, title = 'Feature Importance'):
  feature_importance = model.feature_importances_
  sorted_idx = np.argsort(feature_importance)
  pos = np.arange(sorted_idx.shape[0]) + .5
  fig = plt.figure(figsize=(12, 6))
  plt.barh(pos, feature_importance[sorted_idx], align='center')
  plt.yticks(pos, np.array(feature_name)[sorted_idx])
  plt.title(title)


## Read in collection level data
coll_data = pd.read_csv('collection_level_data.csv')
coll_data.head()

coll_data_ts =  pd.read_csv('collection_level_data_ts.csv')
coll_data_ts.head()

##  Format data type 
## Convert all string fields into category
coll_data['COLLECTION_NAME'] = coll_data['COLLECTION_NAME'].astype('category')
coll_data['ARTIST'] = coll_data['ARTIST'].astype('category')
coll_data['CURATION_STATUS'] = coll_data['CURATION_STATUS'].astype('category')
coll_data['SCRIPT_TYPE'] = coll_data['SCRIPT_TYPE'].astype('category')
coll_data['MINT_CURRENCY'] = coll_data['MINT_CURRENCY'].astype('category')

## Fix different formats in aspect ratio
coll_data['ASPECT_RATIO'] = np.where(coll_data['ASPECT_RATIO']=='1/1', 1, coll_data['ASPECT_RATIO'])
coll_data['ASPECT_RATIO'] = np.where(coll_data['ASPECT_RATIO']=='100/100', 100, coll_data['ASPECT_RATIO'])
coll_data['ASPECT_RATIO'] = coll_data['ASPECT_RATIO'].astype('float64')
print(coll_data.dtypes)

## Drop fields with only 1 category
coll_data = coll_data.drop(['MINT_CURRENCY','IS_DYNAMIC', 'USE_HASH'], axis = 1)

## Prepare modeling data
## Pivot sale price 
coll_price_pvt = coll_data_ts.pivot(index="COLLECTION_NAME", columns="YEAR_MONTH", values="PRICE_USD")
coll_price_pvt.columns=['Dec21_price','Jan21_price','Feb21_price','Mar21_price','Apr21_price','May21_price','Jun21_price','Jul21_price','Aug21_price','Sep21_price'] 

## Pivot sale volume
coll_sale_pvt = coll_data_ts.pivot(index="COLLECTION_NAME", columns="YEAR_MONTH", values="SALE_COUNT")
coll_sale_pvt.columns=['Dec21_sale_num','Jan21_sale_num','Feb21_sale_num','Mar21_sale_num','Apr21_sale_num','May21_sale_num','Jun21_sale_num','Jul21_sale_num','Aug21_sale_num','Sep21_sale_num'] 

## Join coll_data with price monthly change and sale volume monthly change data
coll_data_full = coll_data.merge(coll_price_pvt, on='COLLECTION_NAME', how='left')
coll_data_full = coll_data_full.merge(coll_sale_pvt, on='COLLECTION_NAME', how='left')

## Create dependent variable
y = coll_data_full['AUGUST_SALE_PRICE']
y = y.fillna(0)

## Create independent variables
## August and September are excluded
x = coll_data_full.drop(['COLLECTION_NAME','ARTIST','AUGUST_SALE_COUNT', 'AUGUST_SALE_PRICE','Aug21_price','Sep21_price', 'Aug21_sale_num','Sep21_sale_num'], axis = 1)

## Convert categorical to numeric (LogisticRegression package doesn't take categorical variables)
cleanup_nums = {"CURATION_STATUS":     {"curated": 4, "factory": 2, "playground": 3},
                "SCRIPT_TYPE": {"p5js": 1, "threejs": 2, "js":3, "regl": 4, "zdog": 5, "tonejs": 6, "custom": 7, "a-frame":8, "svg":9 }}

x = x.replace(cleanup_nums)

## Replace n.a. with 999 (can't exclude rows with n.a. because otherwise no Y_c = 0)
x = x.fillna(0)

## Decision Tree Regression
dtr = DecisionTreeRegressor(max_depth = 5, random_state=123).fit(x, y)
dtr_y = dtr.predict(x)
print('The R-squared for Decision Tree regression is: {:.3f}'.format(dtr.score(x, y)))
print('The optimal tree depth after hyper-parameter tuning is: {:.3f}'.format(dtr.get_depth()))

# DOT data
dot_data_dtr = tree.export_graphviz(dtr, out_file=None, 
                                feature_names=x.columns,  
                                class_names = None,
                                filled=True)

# Draw graph
graph_dtr = graphviz.Source(dot_data_dtr, format="png") 
graph_dtr

## Predicted Sale Price
pred_price = pd.DataFrame({"August_Predict_Price":dtr_y})
coll_data_pred = pd.concat([coll_data['COLLECTION_NAME'], round(coll_data['AUGUST_SALE_PRICE']), round(pred_price)], axis = 1)

names = ['Rinascita by Stefano Contiero'
,'Alien Clock by Shvembldr'
,'Return by Aaron Penne'
,'Sentience by pxlq'
,'Eccentrics by Radix'
,'Obicera by Alexis André'
,'Ecumenopolis by Joshua Bagley'
]

predict_s = coll_data_pred.loc[coll_data_pred['COLLECTION_NAME'].isin(names), ["COLLECTION_NAME","August_Predict_Price","AUGUST_SALE_PRICE"]]
predict_s["Undervalue?"] = np.where(predict_s['AUGUST_SALE_PRICE'] < predict_s['August_Predict_Price'], 'Yes', 'No')
predict_s




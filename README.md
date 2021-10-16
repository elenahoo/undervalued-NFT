# undervalued-NFT
 A model that identifies undervalued NFTs.
 
 
This notebook will compare the following 7 collections and assess which collection is undervalued by rank ordering the fair prices predicted from a Decision Tree algorithm. The prediction of the price is based on all collections with similar features and the price they were sold for in the past.

1.   Rinascita by Stefano Contiero
2.   Alien Clock by Shvembldr
3.   Return by Aaron Penne
4.   Sentience by pxlq
5.   Eccentrics by Radix
6.   Obicera by Alexis André
7.   Ecumenopolis by Joshua Bagley

The reason to use Decision Tree model to predict the resale price is based on my previous work [here](https://colab.research.google.com/drive/1HQBG-J9fbNX_G6TWfWQeb9ngtTHC7154?usp=sharing), where other model choices were tested and the conclusion was made based on model performance and interpretability.

Database used in the files is from [Flipside Crypto](https://www.flipsidecrypto.com)

## 1. Data Preparation

The data preparation is the same as in my previoius work where the data is consolidated to a collection level, including the static non-time-series variables and the time-sereis variables. 

* Non-time series data: (`collection_data.csv` | *unique key*: `collection_name`)

* Time-series data (`collection_data_ts.csv` | *unique key*: `collection_name`, `year_month`)

## 2. Decision Tree Regression
Based on all historic information prior to August 1st 2021, the Decision Tree is able to give a tree path for each of the 7 collections and predicts what the potential sale price will be. Given my previous work ([here](https://colab.research.google.com/drive/1HQBG-J9fbNX_G6TWfWQeb9ngtTHC7154)) shows this model gives a very high accuracy, it can be also used to rank order the value of each collection.

The predicted price for each of the 7 collections may not necessarily be the exact price the collection will be sold for, but it's a good indication of where the value of the collection ranks among each other and all other collections.

Since the predicted price is the average price of all collections with similar features (i.e. past sale price, curation status, aspect ratio etc.), if an actual sale price is lower than the this average predicted price, it is deemed as undervalued.

Also, among the 7 collections you can also see the ranking based on the predicted price.

## 3. Conclusion

Using the Decision Tree Regression and its predicted price of collections with similar features, the following 2 collections are undervalued:
- Eccentrics by Radix: similar collections were sold for USD 22k on average in August while it was only sold for USD 13k.
- Sentience by pxlq: similar collections were sold for USD 22k on average in August while it was sold for USD 16k.

Among the 7 collections the following 2 collections have a much lower expected value (USD 3k) than the other 5 (USD 22k):
- Obicera by Alexis Andre
- Rinascita by Stefano Contiero
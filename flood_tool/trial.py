from tool import Tool
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_squared_error, r2_score



new_class = Tool()

df = pd.read_csv('resources/postcodes_sampled_15000.csv') # train model with this 

df2 = pd.read_csv('resources/test_postcodes_5000.csv') # postcodes we want to find 

df3 = pd.read_csv('resources/postcodes_sampled_5000.csv') #true value of those 5000 postcodes 

locations = df2['postcode'].tolist()

y_true = df3['medianPrice']

y_pred = new_class.get_median_house_price_estimate(locations, method=2)

<<<<<<< HEAD
prices = new_class.get_annual_flood_risk(locations)
=======
print(np.sqrt(mean_squared_error(y_true.values, y_pred.values))) 
print(r2_score(y_true.values, y_pred.values))
>>>>>>> d070fedf959f9067c7faf77db05d0ff9fb44d763


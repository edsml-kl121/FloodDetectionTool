"""Example module in template package."""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor


from sklearn.linear_model import SGDClassifier
from sklearn import svm
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


from geo import *

__all__ = ['Tool']

class Tool(object):
    """Class to interact with a postcode database file."""

    def __init__(self, postcode_file='', sample_labels='',
                 household_file=''):

        """
        Parameters
        ----------

        postcode_file : str, optional
            Filename of a .csv file containing geographic location
            data for postcodes.

        sample_labels : str, optional
            Filename of a .csv file containing sample data on property
            values and flood risk labels.

        household_file : str, optional
            Filename of a .csv file containing information on households
            by postcode.
        """
        if postcode_file == '':
            postcode_file = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_unlabelled.csv'))

        if sample_labels == '':
            sample_labels = os.sep.join((os.path.dirname(__file__),
                                         'resources',
                                         'postcodes_sampled.csv'))

        if household_file == '':
            household_file = os.sep.join((os.path.dirname(__file__),
                                          'resources',
                                          'households_per_sector.csv'))
        
        self.label = pd.read_csv(sample_labels)
        self.postcodes = pd.read_csv(postcode_file)
        self.house_label = pd.read_csv(household_file)

        

    def get_easting_northing(self, postcodes):
        """Get a frame of OS eastings and northings from a collection
        of input postcodes.
        Parameters
        ----------
        postcodes: sequence of strs
            Sequence of postcodes.
        Returns
        -------
        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NaN.
         """
        if type(postcodes) is str:
            postcodes = [postcodes]
        postcode_df = self.postcodes
        postcode_df = postcode_df.fillna('np.nan')
        postcode_df_index = postcode_df.set_index('postcode')
        df = pd.DataFrame(columns=(['sector','easting', 'northing','localAuthority']))
        
        for i in range(len(postcodes)):
            if postcodes[i] in postcode_df['postcode'].tolist():
                df.loc[postcodes[i]] = postcode_df_index.loc[postcodes[i]]
            else:
                df.loc[postcodes[i]] = np.NaN
        del df['sector']
        del df['localAuthority']
        
        return df


    def get_lat_long(self, postcodes):
        """Get a frame containing GPS latitude and longitude information for a
        collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input unlabelled postcodes file) return as NAN.
        """
        NE = self.get_easting_northing(postcodes)
        east = NE['easting']
        north = NE['northing']
        lat_long = []
        for i in range(len(NE)):
            postcode = postcodes[i]
            if np.isnan(east[i]):
                lat = np.NaN
                long = np.NaN
            else:
                a = get_gps_lat_long_from_easting_northing([east[i]],[north[i]],rads=False)
                lat = int(a[0])
                long = int(a[1])
            lat_long.append([postcode,lat,long])
            postcode_df = pd.DataFrame(lat_long, columns=('postcode','lat','lon'))
            postcode_df = postcode_df.set_index('postcode')
        return postcode_df


    def get_easting_northing_sample(self, postcodes):
        """Get a frame of OS eastings and northings from a collection
        of input postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only OSGB36 easthing and northing indexed
            by the input postcodes. Invalid postcodes (i.e. not in the
            input sampled postcodes file) return as NaN.
         """
        if type(postcodes) is str:
            postcodes = [postcodes]
        postcode_df = self.label
        postcode_df = postcode_df.fillna('np.nan')
        postcode_df_index = postcode_df.set_index('postcode')
        df = pd.DataFrame(columns=(['sector','easting', 'northing','localAuthority','riskLabel','medianPrice']))
        
        for i in range(len(postcodes)):
            if postcodes[i] in postcode_df['postcode'].tolist():
                df.loc[postcodes[i]] = postcode_df_index.loc[postcodes[i]]
            else:
                df.loc[postcodes[i]] = np.NaN
        del df['sector']
        del df['localAuthority']
        del df['riskLabel']
        del df['medianPrice']
        return df


    def get_lat_long_sample(self, postcodes):
        """Get a frame containing GPS latitude and longitude information for a
        collection of of postcodes.

        Parameters
        ----------

        postcodes: sequence of strs
            Sequence of postcodes.

        Returns
        -------

        pandas.DataFrame
            DataFrame containing only WGS84 latitude and longitude pairs for
            the input postcodes. Invalid postcodes (i.e. not in the
            input sampled postcodes file) return as NAN.
        """
        NE = self.get_easting_northing_sample(postcodes)
        east = NE['easting']
        north = NE['northing']
        lat_long = []
        for i in range(len(NE)):
            postcode = postcodes[i]
            if np.isnan(east[i]):
                lat = np.NaN
                long = np.NaN
            else:
                a = get_gps_lat_long_from_easting_northing([east[i]],[north[i]],rads=False)
                lat = int(a[0])
                long = int(a[1])
            lat_long.append([postcode,lat,long])
            postcode_df = pd.DataFrame(lat_long, columns=('postcode','lat','lon'))
            postcode_df = postcode_df.set_index('postcode')
        return postcode_df

    @staticmethod
    def get_flood_class_methods():
        """
        Get a dictionary of available flood probablity classification methods.

        Returns
        -------

        dict
            Dictionary mapping classification method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_flood_probability method.
        """
        return {'random_forest': 0, "RF_balanced": 1, "SGD_Classifier": 2, "knn": 3, 'GBC': 4}

    def get_flood_class(self, postcodes, method=0):
        """
        Generate series predicting flood probability classification
        for a collection of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_flood_probability_methods) the classification
            method to be used.

        Returns
        -------

        pandas.Series
            Series of flood risk classification labels indexed by postcodes.
        """
        # print("asdas", self.label)
        X = self.label[["easting","northing"]]
        y = self.label['riskLabel']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) # Holdout

        
        northing_eastings = self.get_easting_northing(postcodes)
        # print(northing_eastings, 'asdsa')
        # northing_eastings = X.iloc[0:2]
        # print(self.get_flood_class_methods(), 'asd')
        if method == self.get_flood_class_methods()["random_forest"]:
            model = RandomForestClassifier(criterion = 'gini', max_features = 'log2', 
                                           class_weight = {1: 10, 2: 10, 3: 10, 4: 1, 5: 15, 6: 15, 7: 10, 8: 150, 9: 300, 10: 300})
            model.fit(X_train, y_train)

        if method == self.get_flood_class_methods()["RF_balanced"]:
            over = SMOTE(sampling_strategy='not majority', random_state=41)
            under = RandomUnderSampler(sampling_strategy={1:500}, random_state=43)
            steps = [('u', under)] #, ('o', over)
            pipeline = Pipeline(steps=steps)
            X_train, y_train = pipeline.fit_resample(X_train, y_train)

            model = RandomForestClassifier(criterion = 'gini', max_features = 'log2', 
                                           class_weight = {1: 10, 2: 10, 3: 10, 4: 1, 5: 15, 6: 15, 7: 10, 8: 150, 9: 300, 10: 300})
            model.fit(X_train, y_train)

        if method == self.get_flood_class_methods()["SGD_Classifier"]:
            model = SGDClassifier(loss='hinge', penalty='l1', alpha=1/20)
            model.fit(X_train, y_train)
        if method ==  self.get_flood_class_methods()["knn"]:
            model = KNeighborsClassifier(n_neighbors=20)
            model.fit(X_train, y_train)
        if method ==  self.get_flood_class_methods()["GBC"]:
            model = GradientBoostingClassifier(random_state=1)
            model.fit(X_train, y_train)

        y_new = model.predict(northing_eastings)

        return pd.Series(data=y_new,
                             index=np.asarray(postcodes),
                             name='riskLabel')


    @staticmethod
    def get_house_price_methods():
        """
        Get a dictionary of available flood house price regression methods.

        Returns
        -------

        dict
            Dictionary mapping regression method names (which have
             no inate meaning) on to an identifier to be passed to the
             get_median_house_price_estimate method.
        """

        return {'all_england_median': 0, 'another_1': 1, 'Decision_tree_regressor': 2}

    def get_median_house_price_estimate(self, postcodes, method=2):
        """
        Generate series predicting median house price for a collection
        of poscodes.

        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        method : int (optional)
            optionally specify (via a value in
            self.get_house_price_methods) the regression
            method to be used.

        Returns
        -------

        pandas.Series
            Series of median house price estimates indexed by postcodes.
        """


        df = self.label
        df['outwardDistrict'] = df['postcode'].apply(lambda x: x.split(' ')[0])

        df['sec_num']=df['sector'].apply(lambda x: x.split(' ')[1][0])

        if method == 0:
            return pd.Series(data=np.full(len(postcodes), 245000.0),
                             index=np.asarray(postcodes),
                             name='medianPrice')

       

        elif method == 1:  # another one
            median_price = []
            for code in postcodes:
                
                if code in df['postcode'].values:
                    median_price.append(df[df['postcode']==code]['medianPrice'].values[0])

                elif code.split(' ')[0]+' '+code.split(' ')[1][0] in df['sector'].values:
                    sec = code.split(' ')[0]+' '+code.split(' ')[1][0]
                    median_price.append(df[df['sector'] == sec]['medianPrice'].mean())

                elif code.split(' ')[0] in df['outwardDistrict'].values:
                    district = df[df['outwardDistrict'] == code.split(' ')[0]]
                    X_test = code.split(' ')[1][0]

                    KNN_model = KNeighborsRegressor(n_neighbors=1,weights='distance', n_jobs=-1)
                    X = district[['sec_num']]
                    y = district['medianPrice']
                    KNN_model.fit(X,y)
                    y_pred = KNN_model.predict(pd.DataFrame([X_test], columns=['sec_num']))
                    median_price.append(y_pred[0])
                else:
                    median_price.append(np.nan)
            
            return pd.Series(data= median_price, index = postcodes, name='another_one')
                    

        elif method == 2:  # Decision tree regressor 
            median_price = []
            for code in postcodes:
                
                if code in df['postcode'].values:
                    median_price.append(df[df['postcode']==code]['medianPrice'].values[0])

                elif code.split(' ')[0]+' '+code.split(' ')[1][0] in df['sector'].values:
                    sec = code.split(' ')[0]+' '+code.split(' ')[1][0]
                    median_price.append(df[df['sector'] == sec]['medianPrice'].mean())

                elif code.split(' ')[0] in df['outwardDistrict'].values:
                    district = df[df['outwardDistrict'] == code.split(' ')[0]]

                    X_test = code.split(' ')[1][0] # sector 

                    dtree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=0.13, random_state=3)

                    X = district[['sec_num']]
                    y = district['medianPrice']

                    dtree.fit(X, y)

                    y_pred = dtree.predict(pd.DataFrame([X_test], columns=['sec_num']))
                    median_price.append(y_pred[0])


                else:
                    median_price.append(np.nan)
            
            return pd.Series(data= median_price, index = postcodes, name='another_one')
        
        else:
            raise NotImplementedError

        
        series = pd.Series(median_prices.flatten(), index = postcodes)
        
        return series

    def get_total_value(self, locations):
        """
        Return a series of estimates of the total property values
        of a collection of postcode units or sectors.


        Parameters
        ----------

        locations : sequence of strs
            Sequence of postcode units or sectors


        Returns
        -------

        pandas.Series
            Series of total property value estimates indexed by locations.
        """

        total_price = np.zeros((len(locations), 1), dtype=float)



        for i, string in enumerate(locations): 
            wanted_outwardDistrict = string.split(' ')[0]
            wanted_sector = string.split(' ')[1]
            
            if (len(wanted_outwardDistrict) == 3): # there are double spaces in household_per_sector.csv 
                sector_df = self.house_label[self.house_label['postcode sector'] == wanted_outwardDistrict + '  ' + wanted_sector[0]]
            elif (len(wanted_outwardDistrict) == 4):
                sector_df = self.house_label[self.house_label['postcode sector'] == wanted_outwardDistrict + ' ' + wanted_sector[0]]
            else: # invalid district given 
                total_price[i] == np.nan
            
            
            if len(string) < 7: # is a sector 
                #mean house price in that sector 
                mean_price = self.label[self.label['sector'] == wanted_outwardDistrict + ' ' + wanted_sector[0]]['medianPrice'].mean()

                if len(sector_df != 0): # we have data for that sector 
                    total_price[i] = mean_price * sector_df['households'].item()
                else: # no data for that location 
                    total_price[i] == np.nan
                    
            if len(string) > 6: #is a postcode 
                
                s = [string]
                median_price_series = self.get_median_house_price_estimate(s)
                median_price = median_price_series.item()
                
                
                if len(sector_df != 0): # we have data for that sector
                    total_price[i] = median_price * (sector_df['households'].item()) / (sector_df['number of postcode units'].item())
                else: # no data for that location
                    total_price[i] == np.nan
                


        series = pd.Series(total_price.flatten(), index = locations)
        
        return series

    def get_annual_flood_risk(self, postcodes,  risk_labels=None):
        """
        Return a series of estimates of the total property values of a
        collection of postcodes.

        Risk is defined here as a damage coefficient multiplied by the
        value under threat multiplied by the probability of an event.


        Parameters
        ----------

        postcodes : sequence of strs
            Sequence of postcodes.
        risk_labels: pandas.Series (optional)
            Series containing flood probability classifiers, as
            predicted by get_flood_probability.

        Returns
        -------

        pandas.Series
            Series of total annual flood risk estimates indexed by locations.
        """

        if risk_labels == None: 
            risk_labels = self.get_flood_class(postcodes, method =1 )
        
        total_value = self.get_total_value(postcodes)

        probabilities = np.zeros(len(postcodes))   

        for i, label in enumerate(risk_labels): 
            if label == 1.0: 
                probabilities[i] = 0.05
            if label == 2.0: 
                probabilities[i] = 0.04
            if label == 3.0: 
                probabilities[i] = 0.03
            if label == 4.0: 
                probabilities[i] = 0.02
            if label == 5.0: 
                probabilities[i] = 0.015
            if label == 6.0: 
                probabilities[i] = 0.001
            if label == 7.0: 
                probabilities[i] = 0.005
            if label == 8.0: 
                probabilities[i] = 0.001
            if label == 9.0: 
                probabilities[i] = 0.0005
            if label == 10.0: 
                probabilities[i] = 0.0001

        property_values = total_value.values

        risk = property_values * probabilities

        series = pd.Series(data=risk, index=postcodes)

        return series







# tool = Tool(postcode_file='', sample_labels='', household_file='')

tool = Tool()
# print(tool.get_flood_class_methods())
# print(tool.get_flood_class(['YO62 4LS', 'LN5 7RW', 'CH1 1GZ', 'SL6 3BS','TF9 9DY', 'CV10 7AU', 'CM5 0BE'], 0))

print(tool.get_median_house_price_estimate(['YO62 4LS', 'LN5 7RW', 'SL6 3BS','TF9 9DY', 'CV10 7AU', 'CM5 0BE'], 1))

print(tool.get_flood_class(np.array(tool.postcodes.postcode), 0))


print(Tool().get_easting_northing(['S03 4WN', 'YO62 4LS']))
print(Tool().get_easting_northing('S03 4WN'))
print(Tool().get_lat_long(['S03 4WN', 'YO62 4LS']))
print(Tool().get_easting_northing_sample(['GU1 4XF', 'YO62 4LS']))
print(Tool().get_lat_long_sample(['GU1 4XF', 'YO62 4LS']))


"""Example module in template package."""

import os

import numpy as np
import pandas as pd

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
        self.postcodes = pd.read_csv(postcode_file)
        self.label = pd.read_csv(sample_labels)
        self.household = pd.read_csv(household_file)

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
        postcode_df = self.postcodes
        postcode_df = postcode_df.fillna('np.nan')
        postcode_df = postcode_df.set_index('postcode')
        index_data = postcode_df.loc[postcodes]
        east = np.array(index_data['easting']).T
        north = np.array(index_data['northing']).T
        df = pd.DataFrame(np.vstack((east, north)).transpose(),columns=('east','north'))
        
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
        east = NE['east']
        north = NE['north']
        lat_long = []
        for i in range(len(NE)):
            a = get_gps_lat_long_from_easting_northing([east[i]],[north[i]],rads=False)
            lat_long.append(a)
        return pd.DataFrame(lat_long, columns=('lat','lon'))
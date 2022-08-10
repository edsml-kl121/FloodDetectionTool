# Importing modules

import pandas as pd
import folium
from collections import defaultdict
import branca.colormap
from folium import plugins
import sys
sys.path.append('./flood_tool')
from geo import *
from tool import *
from mapping import *


# Importing data

# The unlabelled postcode data
df1 = pd.read_csv('./flood_tool/resources/postcodes_unlabelled.csv')
# The labelled postcode data
df2 = pd.read_csv('./flood_tool/resources/postcodes_sampled.csv')
# The data on number of households
df3 = pd.read_csv('./flood_tool/resources/households_per_sector.csv')
# The data on measurement stations
df4 = pd.read_csv('./flood_tool/resources/stations.csv')
# The data for a wet day
df5 = pd.read_csv('./flood_tool/resources/wet_day.csv')
# The data for a more typical day
df5 = pd.read_csv('./flood_tool/resources/typical_day.csv')


# Data preprocessing

flood_risk_data = df2.copy()
flood_risk_data = flood_risk_data[["easting", "northing", "riskLabel"]]
flood_risk_data[["col1","col2"]] = flood_risk_data.apply(lambda x: get_gps_lat_long_from_easting_northing([x.easting],[x.northing]), axis=1, result_type='expand')  
flood_risk_data[["latitude","longitude"]] = flood_risk_data.apply(lambda x: (x.col1[0],x.col2[0]), axis = 1, result_type='expand')
flood_risk_data = flood_risk_data.drop(columns=['easting','northing','col1','col2'])

house_data = df2.copy()
house_data = house_data[["easting", "northing", "medianPrice"]]
house_data[["col1","col2"]] = house_data.apply(lambda x: get_gps_lat_long_from_easting_northing([x.easting],[x.northing]), axis=1, result_type='expand')  
house_data[["latitude","longitude"]] = house_data.apply(lambda x: (x.col1[0],x.col2[0]), axis = 1, result_type='expand')
house_data = house_data.drop(columns=['easting','northing','col1','col2'])


# Maps with no markers

def flood_risk_map():
    steps = 10
    colormap = branca.colormap.LinearColormap(colors=['yellow','red'], index=[1,10],vmin=1,vmax=10)
    map = plot_circle(53., 0, 2000.)
    # for i in range(steps):
    #     gradient_map[1/steps*i] = colormap.rgb_hex_str(1/steps*i)

    colormap.add_to(map)

    for loc, p in zip(zip(flood_risk_data["latitude"], flood_risk_data["longitude"]), flood_risk_data["riskLabel"]):
        folium.Circle(
            location=loc,
            radius=10,
            fill=True,
            color=colormap(p),
            fill_opacity=0.7
        ).add_to(map)

    map.add_child(colormap)
    return map

def house_price_map():
    steps = 10
    colormap = branca.colormap.LinearColormap(colors=['yellow','red'], index=[0,1000000],vmin=0,vmax=112409100)
    map = plot_circle(53., 0, 2000.)
    # for i in range(steps):
    #     gradient_map[1/steps*i] = colormap.rgb_hex_str(1/steps*i)

    colormap.add_to(map)

    for loc, p in zip(zip(house_data["latitude"], house_data["longitude"]), house_data["medianPrice"]):
        folium.Circle(
            location=loc,
            radius=10,
            fill=True,
            color=colormap(p),
            fill_opacity=0.7
        ).add_to(map)

    map.add_child(colormap)
    return map



# widget and postcode inputs for risk_floods

def flood_risk_map_pc(postcode):
    tool = Tool()
    steps = 10
    colormap = branca.colormap.LinearColormap(colors=['yellow','red'], index=[1,10],vmin=1,vmax=10)
    map = plot_circle(53., 0, 2000.)
    colormap.add_to(map)

    for loc, p in zip(zip(flood_risk_data["latitude"], flood_risk_data["longitude"]), flood_risk_data["riskLabel"]):
        folium.Circle(
            location=loc,
            radius=10,
            fill=True,
            color=colormap(p),
            fill_opacity=0.7
        ).add_to(map)

    map.add_child(colormap)

#     postcode = 'YO62 4LS'
    position = tool.get_lat_long([postcode])
    flood_risk_series = tool.get_flood_class([postcode], 0)
    flood_value = flood_risk_series.values[0]
    folium.Marker(
      location=[position['lat'], position['lon']],
      popup=f'''postcode: {postcode} <br>
            flood risk: {flood_value}''' ,
            max_width = 400
    ).add_to(map)
    return map

def f_widget_flood(x):
    return flood_risk_map_pc(x)

# widget and postcode inputs for house_price

def house_price_map_pc(postcode):
    steps = 10
    colormap = branca.colormap.LinearColormap(colors=['yellow','red'], index=[0,1000000],vmin=0,vmax=112409100)
    map = plot_circle(53., 0, 2000.)
    tool = Tool()
    # for i in range(steps):
    #     gradient_map[1/steps*i] = colormap.rgb_hex_str(1/steps*i)

    colormap.add_to(map)

    for loc, p in zip(zip(house_data["latitude"], house_data["longitude"]), house_data["medianPrice"]):
        folium.Circle(
            location=loc,
            radius=10,
            fill=True,
            color=colormap(p),
            fill_opacity=0.7
        ).add_to(map)

    map.add_child(colormap)
    position = tool.get_lat_long([postcode])
    house_price_series = tool.get_median_house_price_estimate([postcode], 0)
    price_value = house_price_series.values[0]
    folium.Marker(
      location=[position['lat'], position['lon']],
      popup=f'''postcode: {postcode} <br>
            flood risk: {price_value}''' ,
            max_width = 400 
    ).add_to(map)
    return map

def f_widget_house(x):
    return house_price_map_pc(x)

# interact(f_widget_house, x=widgets.Combobox(options=df1.postcode.tolist(), value="YO62 4LS"));
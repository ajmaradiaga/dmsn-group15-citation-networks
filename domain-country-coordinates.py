#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from fastparquet import ParquetFile
import socket
import geocoder


# In[8]:



pf = ParquetFile('datasets/parquet/domain_aggregation.pq')
df = pf.to_pandas()

countries_df = pd.read_csv('datasets/countries.tsv', sep='\t')


# In[9]:


countries_df.head()


# In[62]:


def retrieve_latlong_from_tld(domain):
    tld = domain.split('.')[-1]
    
    if tld == 'uk':
        tld = 'GB'
    elif tld == 'yu':
        tld = 'RS'
    elif tld == 'su':
        tld = 'RU'
    elif tld == 'cs':
        tld = 'CZ'
    
    if tld != 'bitnet':
        country = countries_df.loc[countries_df['country'] == tld.upper()]
        if country.empty:
            print(f"   Unable to find {tld}")
        else:
            print(f"   Resolved {tld}")
            return {
                'lat': country.iloc[0]['latitude'],
                'lng': country.iloc[0]['longitude'],
                'city': '',
                'country': country.iloc[0]['country']
            }
    
    return None

def format_geocoder_response(g):
    return {
        'lat': g.latlng[0],
        'lng': g.latlng[1],
        'city': g.current_result.city,
        'country': g.current_result.country
    }

def find_geocoder(domain):
    try:
        ip_address = socket.gethostbyname(domain)
        g = geocoder.ip(ip_address)
        return g
    except socket.gaierror as ge:
        print(f"Unable to resolve {domain}")
        return None


# In[63]:


# domain_latlong = {}

for k, v in domain_latlong.items():
    domain = k#row[1][0]
    
    if '.bitnet' not in k and v is None:

        g = find_geocoder(domain)
        if g:
            domain_latlong[domain] = format_geocoder_response(g)
        else:
            new_domain = 'www.' + domain
            g = find_geocoder(new_domain)
            if g:
                print(f"   Resolved {new_domain}")
                domain_latlong[domain] = format_geocoder_response(g)
            else:
                # Last resort, tld
                domain_latlong[domain] = retrieve_latlong_from_tld(domain)
     


# In[54]:


domain = 'ic.ac.uk'
ip_address = socket.gethostbyname(domain)
g = geocoder.ip(ip_address)

g


# In[67]:


import copy
aux = copy.deepcopy(domain_latlong)

for k, v in aux.items():
    if v is None:
        del domain_latlong[k]
        i += 1
        print(k)

print(i)


# In[74]:


domain_latlong_df = pd.DataFrame.from_dict(domain_latlong, orient='index')
domain_latlong_df.head()


# In[75]:


domain_latlong_df.to_csv('datasets/domains_with_countries.csv')


# In[4]:


domain_latlong_df = pd.read_csv('datasets/domains_with_countries.csv')


# In[13]:


country_group_df = domain_latlong_df.groupby(['country']).count().sort_values(['lat'], ascending = False)


# In[14]:


# Contributions by country
country_group_df


# In[ ]:


import datashader as ds
import pandas as pd
from colorcet import fire
from datashader import transfer_functions as tf


# In[ ]:


agg = ds.Canvas().points(domain_latlong_df, 'lat', 'long')
tf.set_background(tf.shade(agg, cmap=fire),"black")


# In[ ]:


import holoviews as hv
import geoviews as gv
from holoviews.operation.datashader import datashade
hv.extension('bokeh')

url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{Z}/{Y}/{X}.jpg'
tile_opts  = dict(width=1000,height=600,xaxis=None,yaxis=None,bgcolor='black',show_grid=False)
map_tiles  = gv.WMTS(url).opts(style=dict(alpha=0.5), plot=tile_opts)
points     = hv.Points(domain_latlong_df, ['lat', 'long'])
taxi_trips = datashade(points, x_sampling=1, y_sampling=1, cmap=fire, width=1000, height=600)

map_tiles * taxi_trips


# In[96]:


from ipyleaflet import Map, basemaps, basemap_to_tiles, Marker, Icon, Polyline, MarkerCluster

center = (52.204793, 0.121558)
center2 = (52.205793, 0.122558)

m = Map(
    layers=(basemap_to_tiles(basemaps.OpenStreetMap.Mapnik), ),
    center=(52.204793, 0.121558),
    zoom=5
)


icon = Icon(icon_url='marker-40.png', icon_size=[40, 40], icon_anchor=[22,94])
# marker1 = Marker(location=center, icon=icon)
#m.add_layer(marker1)

# marker2 = Marker(location=center2, icon=icon)
#m.add_layer(marker2)

for row in country_group_df.iterrows():
    # city = row[0][0]
    country = row[0]
    
    domains = domain_latlong_df[(domain_latlong_df['country'] == country)]
    
    markers = []
    
    for row in domains.iterrows():
        point = (row[1]['lat'], row[1]['lng'])
        markers.append(Marker(location=point, icon=icon))

    marker_cluster = MarkerCluster(
        markers=markers
    )

    m.add_layer(marker_cluster);


# In[ ]:


for row in domain_latlong_df.iterrows():
    point = (row[1]['lat'], row[1]['long'])
    
    marker = Marker(location=point, icon=icon, draggable=False)
    m.add_layer(marker);


# In[ ]:


line = Polyline(
    locations = 
    [[45.51, -122.68],
    [37.77, -122.43]],
    color = "green" ,
    fill_color= "green",
    weight=1,
    stroke=True)
m.add_layer(line)


# In[101]:


m


# In[ ]:





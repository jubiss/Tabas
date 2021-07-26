import pandas as pd

df = pd.read_csv('location.csv')

df['latitude'] = df['latitude'].str.replace(',','.')
df['longitude'] = df['longitude'].str.replace(',','.')
df = df.drop(df[df['latitude'] == '0.00000'].index[0]) #latitude 0, longitude 0
df = df.reset_index()
df = df.drop('index',axis=1)
df['endereço'] = ''

df['bairro'] = ''

lat_long = df[['latitude','longitude']].values

from geopy.geocoders import  GoogleV3

geolocator = GoogleV3(api_key='my_key')

new_info = []
new_locations = []
for i in range(len(df)):
    bairro = -1
#    location = geolocator.reverse(lat_long[i])
    location = geolocator.reverse(df.iloc[i]['latitude']+', '+df.iloc[i]['longitude'])
    for j in location.raw.get('address_components'):
        if 'sublocality' in j.get('types'):
            bairro = j.get('long_name')

    if bairro == -1:
        bairro = location.raw.get('address_components')[0].get('long_name')
    new_info.append([location.address,bairro,location.point])
    new_locations.append(location)
    df.at[i,'endereço'] = location.address
    df.at[i,'bairro'] = bairro

df.to_csv('geo_data.csv')
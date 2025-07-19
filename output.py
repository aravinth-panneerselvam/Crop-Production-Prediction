import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT


# Model Training
# Data Loading
df = pd.read_csv('cleaned_df.csv')

@st.cache_resource
def train_model(df):
    le_country = LabelEncoder()
    le_item = LabelEncoder()

    df['country_encoded'] = le_country.fit_transform(df['country'])
    df['item_encoded'] = le_item.fit_transform(df['item'])

    x = df[['country_encoded', 'item_encoded', 'year', 'area', 'yield']]
    y = df['production']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    rf = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    rf.fit(x_train, y_train)

    return rf, le_country, le_item

rf, le_country, le_item = train_model(df)


#User Interface
st.header('🌎 CROP DISTRIBUTION - COUNTRY WISE')
categories = df['category'].unique()
year = df['year'].unique()
countries = df['country'].unique()

with st.sidebar:
    st.header('Filters')
    selected_category = st.selectbox('category',categories)
    selected_metric = st.selectbox('Metrics',['area','production','yield'])

filtered_df = df[df['category'] == selected_category]

metric_column = selected_metric 
agg_df = filtered_df.groupby('country')[metric_column].sum().reset_index()

agg_df.columns = ['country', f'{selected_metric}  Value']

fig = px.choropleth(
    agg_df,
    locations='country',
    locationmode='country names',
    color= f'{selected_metric}  Value',
    hover_name='country',
    color_continuous_scale='YlOrBr'
)

st.plotly_chart(fig, use_container_width=True)

# SQL Connection with Streamlit to show the table
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def get_connection():
		connection = psycopg2.connect(
			database = 'crop_production_prediction',
			user = 'postgres',
			password = 123456,
			host = 'localhost'
		    )
		connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT) # Set autocommit mode
		return connection

def run_query(sql):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    
    colnames = [desc[0] for desc in cursor.description]
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return pd.DataFrame(rows, columns=colnames)


st.title('🌾 CROP PRODUCTION PREDICTION')

# --------------------CROP TREND ------------------------------------
st.header('Global trends on agri metrics')
st.sidebar.header("Crop Trend")

item_list = sorted(df['item'].unique())
selected_item = st.sidebar.selectbox("Select Crop Item", item_list)

query_list = ['Top 10 countries','Bottom 10 countries']
selected_query = st.sidebar.selectbox("Select Query", query_list)

year_list1 = [2019,2020,2021,2022,2023]
selected_year1 = st.sidebar.selectbox("Year", year_list1)

query = f'''SELECT country,year,SUM({selected_metric}) AS total_{selected_metric} \
    FROM crops_table \
    WHERE item = '{selected_item}' AND year = {selected_year1} \
    GROUP BY country,year \
    ORDER BY total_{selected_metric} DESC \
    LIMIT 10
    '''

# Button to run the query
if st.button("Run Query"):
    try:
        sql_df = run_query(query)
        st.success("✅ Query executed successfully!")
        st.dataframe(sql_df)
    except Exception as e:
        st.error(f"❌ Error: {e}")

# Filter the dataframe by selected item
item_df = df[df['item'] == selected_item]

# Group by year and aggregate
trend_df = item_df.groupby('year')[['production', 'area', 'yield']].sum().reset_index()

# Plot using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'{selected_item} global Year-wise production trends', fontsize=16)

sns.lineplot(data=trend_df, x='year', y='production', ax=axes[0])
axes[0].set_title('Production Trend')

sns.lineplot(data=trend_df, x='year', y='area', ax=axes[1])
axes[1].set_title('Area Trend')

sns.lineplot(data=trend_df, x='year', y='yield', ax=axes[2])
axes[2].set_title('Yield Trend')

metrics = ['production', 'area', 'yield']
titles = ['Production Trend', 'Area Trend', 'Yield Trend']

for i in range(3):
    sns.lineplot(data=trend_df, x='year', y=metrics[i], ax=axes[i])
    axes[i].set_title(titles[i])
    axes[i].xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.tight_layout()
st.pyplot(fig)


st.sidebar.header("Prediction Inputs")

#Prediction
selected_country = st.sidebar.selectbox("Country", sorted(df['country'].unique()))
selected_item = st.sidebar.selectbox("Crop Item", sorted(df['item'].unique()))

country_encoded = le_country.transform([selected_country])[0]
item_encoded = le_item.transform([selected_item])[0]


if 'model' not in st.session_state:
    rf, le_country, le_item = train_model(df)
    st.session_state['model'] = rf
    st.session_state['le_country'] = le_country
    st.session_state['le_item'] = le_item
else:
    rf = st.session_state['model']
    le_country = st.session_state['le_country']
    le_item = st.session_state['le_item']


col1, col2, col3 = st.columns(3)

with col1:
    area = st.number_input("Area in hectares", min_value=0.0)

with col2:
    yield1 = st.number_input("Yield in Kg/hectare", min_value=0.0)

with col3:
    # year_list = df['Year'].unique()
    year_list = [2019,2020,2021,2022,2023,2024, 2025, 2026, 2027, 2028]
    selected_year = st.selectbox("Year", sorted(year_list))

if st.button("🔮 Predict"):
    with st.spinner("Predicting..."):
        y_predict = rf.predict([[country_encoded, item_encoded, selected_year, area, yield1]])
        st.success(f'The production prediction is: {y_predict[0]:,.2f} tonnes')

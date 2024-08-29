import folium
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder


def get_clean_data():
    """
    Load and clean crime data from CSV.

    Returns:
        pd.DataFrame: Cleaned dataframe with necessary columns and encoded 'primary_type' column.
        LabelEncoder: LabelEncoder used to encode 'primary_type'.
    """

    dtype = {
        'beat': 'int32',
        'district': 'int16',
        'ward': 'int16',
        'community_area': 'int16',
        'year': 'int16',
        'latitude': 'float32',
        'longitude': 'float32'
    }

    pd.set_option('display.max_columns', None)
    df = pd.read_csv('../clean_df.csv')
    label_encoder = LabelEncoder()
    df['primary_type'] = label_encoder.fit_transform(df['primary_type'])
    return df, label_encoder

def add_sidebar():
    """
    Add a sidebar to the Streamlit app for user inputs.

    Returns:
        dict: Dictionary with selected options for hours and crime types.
    """
    st.sidebar.header('Predict the next hotspot now!')
    input_dict = {}
    with st.sidebar.form("my_form"):
        input_dict['hours'] = st.slider(
        label='Select the next hours you want to predict',
        min_value=1,
        max_value=24,
        value=24
        )
        # Customize crimes types prediction
        crime_options = [
            'THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT', 'ASSAULT', 'DECEPTIVE PRACTICE',
            'OTHER OFFENSE', 'ROBBERY', 'WEAPONS VIOLATION', 'BURGLARY', 'NARCOTICS', 'CRIMINAL TRESPASS',
            'OFFENSE INVOLVING CHILDREN', 'CRIMINAL SEXUAL ASSAULT', 'SEX OFFENSE', 'PUBLIC PEACE VIOLATION',
            'HOMICIDE', 'INTERFERENCE WITH PUBLIC OFFICER', 'STALKING', 'ARSON', 'PROSTITUTION', 'INTIMIDATION',
            'LIQUOR LAW VIOLATION', 'CONCEALED CARRY LICENSE VIOLATION', 'KIDNAPPING', 'OBSCENITY', 'GAMBLING',
            'HUMAN TRAFFICKING', 'PUBLIC INDECENCY', 'OTHER NARCOTIC VIOLATION', 'NON-CRIMINAL'
        ]
        input_dict['crime_types'] = st.multiselect(
            label='Select the types of crime',
            options=crime_options,
            default=crime_options  # All options selected by default
        )
        submitted = st.form_submit_button("Predict")
        return input_dict, submitted


def Generate_future_dates(hours):
    """
    Generate future dates for the specified number of hours.

    Args:
        hours (int): Number of hours to forecast.

    Returns:
        pd.DataFrame: DataFrame with future dates and empty 'primary_type' column.
    """
    current_time = datetime.now()
    future_dates = [current_time + timedelta(hours=x) for x in range(hours)]
    
    # Prepare data for Random Forest predictions
    forecast_data = pd.DataFrame({
        'hour': [date.hour for date in future_dates],
        'day': [date.day for date in future_dates],
        'month': [date.month for date in future_dates],
        'day_of_week': [date.weekday() for date in future_dates],
        'primary_type': np.nan  # We will predict 'Primary Type' later
    })
    return forecast_data

def prepare_data_to_visualization(hours, forecast_data, forecast):
    """
    Prepare simulated data for visualization based on forecasted data.

    Args:
        hours (int): Number of hours for the forecast.
        forecast_data (pd.DataFrame): DataFrame containing forecast data.
        forecast (pd.Series): Forecasted counts of crimes.

    Returns:
        pd.DataFrame: DataFrame containing simulated crime data for visualization.
    """
    simulated_data = []
    for i in range(hours):
        forecast_count = int(forecast.iloc[i])
        if forecast_count > 0:
            for _ in range(forecast_count):
                simulated_data.append({
                    'latitude': forecast_data['latitude'].iloc[i],
                    'longitude': forecast_data['longitude'].iloc[i],
                    'hour': forecast_data['hour'].iloc[i],
                    'day': forecast_data['day'].iloc[i],
                    'month': forecast_data['month'].iloc[i],
                    'day_of_week': forecast_data['day_of_week'].iloc[i],
                    'primary_type': forecast_data['primary_type'].iloc[i]
                })
    
    simulated_df = pd.DataFrame(simulated_data)
    return simulated_df

def hotspot_visualization(input_data, forecast_data, simulated_df):
    """
    Create and display a hotspot map with crime predictions.

    Args:
        input_data (dict): User inputs from the sidebar.
        forecast_data (pd.DataFrame): DataFrame with forecasted data.
        simulated_df (pd.DataFrame): DataFrame with simulated crime data.
    """
    # each crime has color in the map
    color_map = {
        'THEFT': 'red',
        'BATTERY': 'blue',
        'CRIMINAL DAMAGE': 'green',
        'MOTOR VEHICLE THEFT': 'purple',
        'ASSAULT': 'orange',
        'DECEPTIVE PRACTICE': 'darkred',
        'OTHER OFFENSE': 'lightred',
        'ROBBERY': 'darkblue',
        'WEAPONS VIOLATION': 'darkgreen',
        'BURGLARY': 'cadetblue',
        'NARCOTICS': 'cyan',
        'CRIMINAL TRESPASS': 'magenta',
        'OFFENSE INVOLVING CHILDREN': 'lime',
        'CRIMINAL SEXUAL ASSAULT': 'coral',
        'SEX OFFENSE': 'teal',
        'PUBLIC PEACE VIOLATION': 'salmon',
        'HOMICIDE': 'gold',
        'INTERFERENCE WITH PUBLIC OFFICER': 'indigo',
        'STALKING': 'violet',
        'ARSON': 'azure',
        'PROSTITUTION': 'peachpuff',
        'INTIMIDATION': 'khaki',
        'LIQUOR LAW VIOLATION': 'sandybrown',
        'CONCEALED CARRY LICENSE VIOLATION': 'tomato',
        'KIDNAPPING': 'orangered',
        'OBSCENITY': 'darkred',
        'GAMBLING': 'olive',
        'HUMAN TRAFFICKING': 'maroon',
        'PUBLIC INDECENCY': 'darkviolet',
        'OTHER NARCOTIC VIOLATION': 'darkslateblue',
        'NON-CRIMINAL': 'lightgrey'
    }
    data, label_encoder = get_clean_data()
    primry_type_label_mapping = dict(enumerate(label_encoder.classes_))
    forecast_data['primary_type'] = [primry_type_label_mapping[label] for label in forecast_data['primary_type']]
    simulated_df['primary_type'] = [primry_type_label_mapping[label] for label in simulated_df['primary_type']]
    map_center = [41.8781, -87.6298]  # Latitude and Longitude for Chicago
    hotspot_map = folium.Map(location=map_center, zoom_start=12)

    for crime_type in input_data['crime_types']:
        crime_data = simulated_df[simulated_df['primary_type'] == crime_type]
        for _, row in crime_data.iterrows():
            tooltip_text = f"Crime: {row['primary_type']}, Hour: {row['hour']}"
            folium.CircleMarker(
                location=(row['latitude'], row['longitude']),
                radius=7,
                color=color_map[crime_type],
                fill=True,
                fill_color=color_map[crime_type],
                fill_opacity=0.7,
                tooltip=tooltip_text  
            ).add_to(hotspot_map)
    # Store the map in session state
    st.session_state['hotspot_map'] = hotspot_map
    # Display the map in Streamlit
    st_folium(hotspot_map, width=800, height=600)

def predict_hotspot_crimes(input_data):
    """
    Predict hotspot crimes using pre-trained models and visualize results.

    Args:
        input_data (dict): User inputs from the sidebar.
    """
    time_series_model = joblib.load('../models/time_series.pkl')
    longitude_and_latitude_model = joblib.load('../models/longitude_and_latitude.pkl')
    primary_type_model = joblib.load('../models/primary_type.pkl')
    
    forecast = time_series_model.forecast(steps=input_data['hours'])
    forecast_data = Generate_future_dates(input_data['hours'])
    forecast_data['primary_type'] = primary_type_model.predict(forecast_data[['hour', 'day', 'month', 'day_of_week']])
    forecast_lat_and_long = longitude_and_latitude_model.predict(forecast_data[['hour', 'day', 'month', 'day_of_week', 'primary_type']])
    forecast_data['latitude'] = forecast_lat_and_long[:, 0]
    forecast_data['longitude'] = forecast_lat_and_long[:, 1]
    
    simulated_df = prepare_data_to_visualization(input_data['hours'], forecast_data, forecast)
    hotspot_visualization(input_data, forecast_data, simulated_df)


def main():
    """
    Main function to run the Streamlit app.

    This function sets up the Streamlit application:
    - Displays the app title and description.
    - Adds a sidebar for user input.
    - Calls the prediction and visualization function when the user clicks the 'Predict' button.
    """

    st.set_page_config(
        layout='wide',
        page_title='Crimes Hotspot',
        page_icon=':round_pushpin:',
        initial_sidebar_state='expanded')
    
    input_data, submitted = add_sidebar()
    
    with st.container():
        st.title('Crime Hotspot Prediction')
        st.write('This project predicts crime hotspots for the next 24 hours by forecasting the likely locations \
                (latitude and longitude), types of crimes (e.g., theft, assault), and the expected number of crimes \
                per hour. By combining these predictions, the system helps identify where and when crimes might occur, \
                enabling law enforcement to take proactive measures to prevent crimes and improve community safety.')
        
        # if the user click on predict 
        if submitted:
            predict_hotspot_crimes(input_data)
        
        # Check if the map exists in session state and display it
        if 'hotspot_map' not in st.session_state or submitted:
            # show the default map for all crimes in the next 24 hours when the app loads
            if not submitted:
                input_data = {
                    'hours': 24,
                    'crime_types': [
                        'THEFT', 'BATTERY', 'CRIMINAL DAMAGE', 'MOTOR VEHICLE THEFT', 'ASSAULT', 'DECEPTIVE PRACTICE',
                        'OTHER OFFENSE', 'ROBBERY', 'WEAPONS VIOLATION', 'BURGLARY', 'NARCOTICS', 'CRIMINAL TRESPASS',
                        'OFFENSE INVOLVING CHILDREN', 'CRIMINAL SEXUAL ASSAULT', 'SEX OFFENSE', 'PUBLIC PEACE VIOLATION',
                        'HOMICIDE', 'INTERFERENCE WITH PUBLIC OFFICER', 'STALKING', 'ARSON', 'PROSTITUTION', 'INTIMIDATION',
                        'LIQUOR LAW VIOLATION', 'CONCEALED CARRY LICENSE VIOLATION', 'KIDNAPPING', 'OBSCENITY', 'GAMBLING',
                        'HUMAN TRAFFICKING', 'PUBLIC INDECENCY', 'OTHER NARCOTIC VIOLATION', 'NON-CRIMINAL'
                    ]
                }
            predict_hotspot_crimes(input_data)
        # display the map that predicted
        if 'hotspot_map' in st.session_state:
            st_folium(st.session_state['hotspot_map'], width=800, height=600)

        #colors theme for charts 
        colors = [
            '#FFCCCC',  # Light Pinkish Red
            '#FF9999',  # Soft Red
            '#FF6666',  # Lighter Red
            '#FF4C4C',  # Light Red
            '#FF3333',  # Medium Red
            '#FF1A1A',  # Bright Red
            '#FF0000',  # Pure Red
            '#E60000',  # Strong Red
            '#CC0000',  # Dark Red
            '#B30000',  # Deep Red
            '#990000',  # Very Dark Red
            '#800000'   # Maroon 
            ]
        df,label_encoder = get_clean_data()
        primry_type_label_mapping = dict(enumerate(label_encoder.classes_))
        df['primary_type'] = [primry_type_label_mapping[label] for label in df['primary_type']]

        st.subheader('Crime Data Analysis')
        col1 , col2 = st.columns(2)
        with col1:
            arrest_rate = df['arrest'].value_counts(normalize=True)
            fig = go.Figure(data=[go.Pie(labels=arrest_rate.index, 
            values=arrest_rate.values, marker=dict(colors=colors))])
            fig.update_layout(title_text='Percentage of Crimes Resulting in Arrests',width=300,height=300)
            fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.plotly_chart(fig)

        with col2:
            arrest_rate_domestic = df['domestic'].value_counts() * 100
            fig = go.Figure(data=[go.Pie(labels=arrest_rate_domestic.index, 
            values=arrest_rate_domestic.values, marker=dict(colors=colors))])
            fig.update_layout(title_text='Number of Domestic Crimes Out of Total Crimes',width=300,height=300)
            fig.update_traces(hoverinfo='label+percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
            st.plotly_chart(fig)


        crime_by_hour = df['hour'].value_counts().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crime_by_hour.index, y=crime_by_hour.values, mode='lines+markers', line=dict(color=colors[2])))
        fig.update_layout(
            title_text='Crime Frequency by Time of Day',
            xaxis_title='Hour',
            yaxis_title='Number of Crimes',
            template='plotly_white'
        )
        st.plotly_chart(fig)

        crime_by_day = df['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crime_by_day.index, y=crime_by_day.values, mode='lines+markers', line=dict(color=colors[2])))
        fig.update_layout(
            title_text='Crime Frequency by Day of the Week',
            xaxis_title='Day of the Week',
            yaxis_title='Number of Crimes',
            template='plotly_white'
        )
        st.plotly_chart(fig)

        crime_by_month = df['month'].value_counts().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=crime_by_month.index, y=crime_by_month.values, mode='lines+markers', line=dict(color=colors[2])))
        fig.update_layout(
            title_text='Crime Frequency by Month',
            xaxis_title='Month',
            yaxis_title='Number of Crimes',
            template='plotly_white'
        )
        st.plotly_chart(fig)

        col1 , col2 = st.columns(2)
        with col1:
            crime_types = df['primary_type'].value_counts().reset_index()
            crime_types.columns = ['primary_type', 'count']
            fig = px.bar(crime_types, 
                        x='primary_type', 
                        y='count', 
                        title='Most Common Types of Crimes')

            fig.update_traces(marker_color=colors[:len(crime_types)], 
                            marker_line_color='#000000',
                            marker_line_width=2, 
                            opacity=0.6)
            fig.update_layout(xaxis_title='',
                            yaxis_title='Count', 
                            template='plotly_white',
                            width=300,height=300)

            st.plotly_chart(fig)

        with col2:
            location_crime_counts = df['location_description'].value_counts().head(15).reset_index()
            location_crime_counts.columns = ['location_description', 'count']
            fig = px.bar(location_crime_counts, 
                        y='count', 
                        title='Top 15 Locations with the Highest Crime Rates')

            fig.update_traces(marker_color=colors[:len(crime_types)], 
                            marker_line_color='#000000',
                            marker_line_width=2, 
                            opacity=0.6)
            fig.update_layout(yaxis_title='Count', 
                            # showticklabels=False,
                            template='plotly_white',
                            width=300,height=300)

            st.plotly_chart(fig)
        col1 , col2 = st.columns(2)
        with col1:
            district_crime_counts = df['district'].value_counts().head(10).reset_index()
            district_crime_counts.columns = ['districts', 'count']
            fig = px.bar(district_crime_counts, 
                        x='districts', 
                        y='count', 
                        title='Top 10 Districts with the Highest Crime Rates',
                        color_continuous_scale='reds')

            fig.update_traces(marker_color=colors, 
                            marker_line_color='#000000',
                            marker_line_width=2, 
                            opacity=0.6)
            fig.update_layout(xaxis_title='District', 
                            yaxis_title='Count', 
                            template='plotly_white',
                            width=300,height=300)

            st.plotly_chart(fig)
        
        with col2:
            community_crime_counts = df['community_area'].value_counts().head(10).reset_index()
            community_crime_counts.columns = ['community_area', 'count']
            fig = px.bar(community_crime_counts, 
                        x='community_area', 
                        y='count', 
                        title='Top 10 Community Areas with the Highest Crime Rates')

            fig.update_traces(marker_color=colors, 
                            marker_line_color='#000000',
                            marker_line_width=2, 
                            opacity=0.6)
            fig.update_layout(xaxis_title='Community Area', 
                            yaxis_title='Count', 
                            template='plotly_white',
                            width=300,height=300)

            st.plotly_chart(fig)
        
        fig = px.scatter(df, 
                        x='longitude', 
                        y='latitude', 
                        color='primary_type', 
                        title='Crime Hotspots',
                        opacity=0.5,
                        color_continuous_scale='reds')

        fig.update_layout(
            title_text='Crime Hotspots',
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            template='plotly_white'
        )
        st.plotly_chart(fig)
        

if __name__ == '__main__':
    main()
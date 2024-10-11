import pickle
import streamlit as st
import pandas as pd

pickle_in = open('model.pkl','rb')
model,scaler,encoder = pickle.load(pickle_in)


def prediction(dataframe):
    predict = model.predict(dataframe)
    return predict

def main():
    st.title('Rain Prediction')

    mintemp = st.number_input('Min Temp', value=None, min_value=0.0)
    maxtemp = st.number_input('Max Temp',value=None, min_value=0.0)
    rainfall = st.number_input('Rainfall',value=None, min_value=0.0)
    WindGustDir = st.selectbox('Wind Gust Direction',options=['W', 'WNW', 'WSW', 'NE', 'NNW', 'N', 'NNE', 'SW', 'ENE', 'SSE',
       'S', 'NW', 'SE', 'ESE', 'E', 'SSW'])
    WindGustSpeed = st.number_input('Gust Speed',value=None, min_value=0.0)
    WindDir9am = st.selectbox('Direction of Wind at 9 am',options=['W', 'NNW', 'SE', 'ENE', 'SW', 'SSE', 'S', 'NE', 'SSW', 'N', 'WSW',
       'ESE', 'E', 'NW', 'WNW', 'NNE'])
    WindDir3pm = st.selectbox('Direction of wind at 3pm',options=['WNW', 'WSW', 'E', 'NW', 'W', 'SSE', 'ESE', 'ENE', 'NNW', 'SSW',
       'SW', 'SE', 'N', 'S', 'NNE', 'NE'])
    WindSpeed9am = st.number_input('Speed of Wind at 9 am',value=None, min_value=0.0)
    WindSpeed3pm = st.number_input('Speed of Wind at 3 pm', value=None, min_value=0.0)
    Humidity9am = st.number_input('Humidity at 9 am', value=None, min_value=0.0)
    Humidity3pm = st.number_input('Humidity at 3 pm', value=None, min_value=0.0)
    Pressure9am = st.number_input('Pressure at 9 am', value=None, min_value=0.0)
    Pressure3pm = st.number_input('Pressure at 3 pm', value=None, min_value=0.0)
    cloud9am = st.number_input('Cloud at 9 am', value=None, min_value=0.0)
    cloud3pm = st.number_input('Cloud at 3 pm', value=None, min_value=0.0)
    temp9am = st.number_input('Temperature at 9 am', value=None, min_value=0.0)
    temp3pm = st.number_input('Temperature at 3 pm', value=None, min_value=0.0)
    raintoday = st.selectbox('Did it rain today', options=['Yes','No'])

    data = pd.DataFrame({
        'MinTemp' : [mintemp], 'MaxTemp' : [maxtemp], 'Rainfall' : [rainfall], 'WindGustDir' :[WindGustDir],
       'WindGustSpeed' : [WindGustSpeed], 'WindDir9am' : [WindDir9am], 'WindDir3pm' : [WindDir3pm], 'WindSpeed9am' : [WindSpeed9am],
       'WindSpeed3pm' : [WindSpeed3pm], 'Humidity9am' : [Humidity9am], 'Humidity3pm' : [Humidity3pm], 'Pressure9am' : [Pressure9am],
       'Pressure3pm' : [Pressure3pm], 'Cloud9am' : [cloud9am], 'Cloud3pm' : [cloud3pm], 'Temp9am' : [temp9am], 'Temp3pm': [temp3pm],
       'RainToday' : [raintoday]
    })
    cat_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
    one_hot_values = encoder.transform(data[cat_columns])
    one_hot_df = pd.DataFrame(one_hot_values, columns=encoder.get_feature_names_out(cat_columns))
    data = pd.concat([data.reset_index(drop=True), one_hot_df.reset_index(drop=True)],axis=1)
    data.drop(columns=cat_columns,inplace=True)

    scaled_data = scaler.transform(data)

    if st.button("Predict"):
        result = prediction(dataframe=scaled_data)
        price = f'Price: {result[0]}'
        st.write(price)



if __name__ == '__main__':
    main()

    


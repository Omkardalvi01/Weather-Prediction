from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import pickle
import pandas as pd

# Load your model, scaler, and encoder
pickle_in = open('model.pkl','rb')
model, scaler, encoder = pickle.load(pickle_in)

# creating the Flask app
app = Flask(__name__)
# creating an API object
api = Api(app)

# Endpoint for predictions
class PredictRain(Resource):
    def post(self):
        try:
            data = request.get_json()
            
            # Convert incoming data to a pandas DataFrame
            df = pd.DataFrame([data])
            cat_columns = ['WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday']
            one_hot_values = encoder.transform(df[cat_columns])
            one_hot_df = pd.DataFrame(one_hot_values, columns=encoder.get_feature_names_out(cat_columns))
            df = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
            df.drop(columns=cat_columns, inplace=True)

            # Scale the data
            scaled_data = scaler.transform(df)
            
            # Predict using the model
            prediction = model.predict(scaled_data)
            
            # Return the result
            return jsonify({'prediction': int(prediction[0])})
        
        except Exception as e:
            return jsonify({'error': str(e)})

# adding the defined resource along with its corresponding url
api.add_resource(PredictRain, '/predict')

# driver function
if __name__ == '__main__':
    app.run(debug=True)

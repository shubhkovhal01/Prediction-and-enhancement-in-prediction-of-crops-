# Importing essential libraries and modules

from flask import Flask, jsonify, redirect, render_template, request, Markup
import numpy as np
import pandas as pd

from utils.fertilizer import fertilizer_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9


#==========================================================================================


#creating instance
app=Flask(__name__)

model=pickle.load(open('RandomForest.pkl','rb'))
model_pro=pickle.load(open('model.pkl','rb'))


# Define the list of features in the same order as during training
model_features = ['Crop', 'District_Name', 'State_Name', 'Area', 'Season']


#===========================================================================================

# Loading plant disease classification model

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                       'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
#===================================================================================================


disease_model_path = 'plant-disease-model.pth'
disease_model = ResNet9(3, len(disease_classes))
state_dict = torch.load(disease_model_path, map_location=torch.device('cpu'))
disease_model.load_state_dict(state_dict)
disease_model.eval()
 





# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

#===================================================================================

def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

######################################################################################################################################################


# render home page

@ app.route('/')
def home():
    title = 'Agropro- Home'
    return render_template('index.html', title=title)


# render crop recommendation form page

@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Agropro- Crop Recommendation'
    return render_template('crop.html', title=title)


# render fertilizer recommendation form page

@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Agropro - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)


@ app.route('/crop_production')
def crop_production():
    title = 'Agropro - Crop Production Prediction'
    return render_template('crop_production.html', title=title)


#=====================================================================================

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Agro - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)
 
#=======================================================================================
 # render crop production result pagemodel_features = ['Crop', 'District_Name', 'State_Name', 'Area', 'Season']

state_values =['Maharashtra']
crop_values = ['Arecanut', 'Other Kharif pulses', 'Rice', 'Banana', 'Cashewnut', 'Coconut', 'Dry ginger', 'Sugarcane', 'Sweet potato', 'Tapioca', 'Black pepper', 'Dry chillies', 'other oilseeds', 'Turmeric', 'Maize', 'Moong(Green Gram)', 'Urad', 'Arhar/Tur', 'Groundnut', 'Sunflower', 'Bajra', 'Castor seed', 'Cotton(lint)', 'Horse-gram', 'Jowar', 'Korra', 'Ragi', 'Tobacco', 'Gram', 'Wheat', 'Masoor', 'Sesamum', 'Linseed', 'Safflower', 'Onion', 'other misc. pulses', 'Samai', 'Small millets', 'Coriander', 'Potato', 'Other Rabi pulses', 'Soyabean', 'Beans & Mutter(Vegetable)', 'Bhindi', 'Brinjal', 'Citrus Fruit', 'Cucumber', 'Grapes', 'Mango', 'Orange', 'other fibres', 'Other Fresh Fruits', 'Other Vegetables', 'Papaya', 'Pome Fruit', 'Tomato', 'Rapeseed &Mustard', 'Mesta', 'Cowpea(Lobia)', 'Lemon', 'Pome Granet', 'Sapota', 'Cabbage', 'Peas (vegetable)', 'Niger seed', 'Bottle Gourd', 'Sannhamp', 'Varagu', 'Garlic', 'Ginger', 'Oilseeds total', 'Pulses total', 'Jute', 'Peas & beans (Pulses)', 'Blackgram', 'Paddy', 'Pineapple', 'Barley', 'Khesari', 'Guar seed', 'Moth', 'Other Cereals & Millets', 'Cond-spcs other', 'Turnip', 'Carrot', 'Redish', 'Arcanut (Processed)', 'Atcanut (Raw)', 'Cashewnut Processed', 'Cashewnut Raw', 'Cardamom', 'Rubber', 'Bitter Gourd', 'Drum Stick', 'Jack Fruit', 'Snak Guard', 'Pump Kin', 'Tea', 'Coffee', 'Cauliflower', 'Other Citrus Fruit', 'Water Melon', 'Total foodgrain', 'Kapas', 'Colocosia', 'Lentil', 'Bean', 'Jobster', 'Perilla', 'Rajmash Kholar', 'Ricebean (nagadal)', 'Ash Gourd', 'Beet Root', 'Lab-Lab', 'Ribed Guard', 'Yam', 'Apple', 'Peach', 'Pear', 'Plums', 'Litchi', 'Ber', 'Other Dry Fruit', 'Jute & mesta']
district_values = ['Ahmednagar', 'Akola', 'Amravati', 'Aurangabad', 'Beed',
       'Bhandara', 'Buldhana', 'Chandrapur', 'Dhule', 'Gadchiroli',
       'Gondia', 'Hingoli', 'Jalgaon', 'Jalna', 'Kolhapur', 'Latur',
       'Mumbai', 'Nagpur', 'Nanded', 'Nandurbar', 'Nashik', 'Osmanabad',
       'Palghar', 'Parbhani', 'Pune', 'Raigad', 'Ratnagiri', 'Sangli',
       'Satara', 'Sindhudurg', 'Solapur', 'Thane', 'Wardha', 'Washim',
       'Yavatmal']
season_values = ['Kharif', 'Whole Year', 'Autumn', 'Rabi', 'Summer', 'Winter']


from sklearn.preprocessing import LabelEncoder

# Load and initialize the label encoders
crop_encoder = LabelEncoder()
district_encoder = LabelEncoder()
state_encoder = LabelEncoder()
season_encoder = LabelEncoder()



# Fit the label encoders with the appropriate mappings
crop_encoder.fit(crop_values)  # crop_values should be a list/array of all possible crop values
district_encoder.fit(district_values)  # district_values should be a list/array of all possible district values
state_encoder.fit(state_values)  # state_values should be a list/array of all possible state values
season_encoder.fit(season_values)  # season_values should be a list/array of all possible season values


@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    crop = request.form.get('crop')
    district = request.form.get('district')
    state = request.form.get('state')
    area = float(request.form.get('area'))
    season = request.form.get('season')

    # Encode the categorical variables
    crop_encoded = crop_encoder.transform([crop])[0]
    district_encoded = district_encoder.transform([district])[0]
    state_encoded = state_encoder.transform([state])[0]
    season_encoded = season_encoder.transform([season])[0]

    # Make the prediction
    features = [crop_encoded, district_encoded, state_encoded, area, season_encoded]
    prediction = model_pro.predict([features])[0]


    # Return the prediction as a response
    return render_template('crop_production-result.html', prediction='The predicted production is {:.2f} tons.'.format(prediction))



#=========================================================================================
# render fertilizer recommendation result page


@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Agropro - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

#==================================================================================================

# render disease prediction result page


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

#=======================================================================================================


if __name__ == '__main__':
    app.run(debug=True)






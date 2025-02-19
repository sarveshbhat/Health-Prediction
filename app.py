from functools import wraps
import string
import bcrypt
from flask import Flask, redirect, render_template, jsonify, request, session, url_for
import joblib
import pandas as pd
import numpy as np
from joblib import load
from pymongo import MongoClient
import requests

app = Flask(__name__)
headers = {
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNjY1NWIwZWMtZjkzMy00MDc2LWE4NzgtNjEzOWNiYTk1MTkwIiwidHlwZSI6ImFwaV90b2tlbiJ9.7ieYfg4l_mq5neAcYnKExGVKM-6VeLxmpM22zJGYPCg"
           }
app.secret_key = 'b\xb8\xc18\x8f/\xd8\x97\xc0c[\xb5zd\xe8\x1f\xa4\xbc\xa5\x1c\xec\x86\xa5?\xb1'
client = MongoClient('mongodb://localhost:27017/')
db = client['DiseasePrediction']
users_collection = db['users']   # Collection for session management



filename = 'diabetes-prediction-rfc-model.pkl'
classifier = joblib.load(filename)


# Load pre-trained model
clf = load("./saved_model/decision_tree.joblib")

# Define symptoms
symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
            'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
            'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
            'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
            'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
            'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
            'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
            'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0, 'mild_fever': 0,
            'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0, 'swelling_of_stomach': 0,
            'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0, 'throat_irritation': 0,
            'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0, 'weakness_in_limbs': 0,
            'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
            'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0, 'swollen_legs': 0,
            'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0, 'swollen_extremeties': 0,
            'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
            'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0, 'movement_stiffness': 0,
            'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0, 'loss_of_smell': 0,
            'bladder_discomfort': 0, 'foul_smell_of urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0, 'internal_itching': 0,
            'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
            'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0, 'watering_from_eyes': 0,
            'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0, 'lack_of_concentration': 0,
            'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
            'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0, 'fluid_overload.1': 0,
            'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0, 'pus_filled_pimples': 0,
            'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0, 'inflammatory_nails': 0,
            'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0}

url = "https://api.edenai.run/v2/text/question_answer"

def get_disease_description(disease):
    question = "What are the details of the predicted disease?"
    context = f"The predicted disease is {disease} i need in senetnces."
    
    payload = {
        "providers": "openai",
        "texts": [context],
        "question": question,
        "examples_context": "In 2017, U.S. life expectancy was 78.6 years.",
        "examples": [["What is human life expectancy in the United States?", "78 years."]],
        "fallback_providers": ""
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result['openai']['answers'][0]
    else:
        return "Sorry, something went wrong."


@app.route('/major-hospitals', methods=['POST'])
def get_major_hospitals():
    data = request.get_json()
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    # Use Overpass API to fetch major hospitals in a larger radius
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      node["amenity"="hospital"](around:20000,{latitude},{longitude});
      way["amenity"="hospital"](around:20000,{latitude},{longitude});
      relation["amenity"="hospital"](around:20000,{latitude},{longitude});
    );
    out center;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    hospitals = []
    for element in data['elements']:
        if 'tags' in element and 'name' in element['tags']:
            hospital = {
                'name': element['tags']['name'],
                'lat': element['lat'] if 'lat' in element else element['center']['lat'],
                'lon': element['lon'] if 'lon' in element else element['center']['lon']
            }
            hospitals.append(hospital)
    number_of_hospital=20
    hospitals = sorted(hospitals, key=lambda k: (float(k['lat']) - float(latitude))**2 + (float(k['lon']) - float(longitude))**2)[:number_of_hospital]
    return jsonify({'hospitals': hospitals})


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['username']
        email = request.form['email']
        mobile = request.form['mobile']
        password1 = request.form['password']
        password2 = request.form['confirm_password']

        # Check if passwords match
        if password1 != password2:
            return render_template('signup.html', message='Passwords do not match')

        # Check if password meets criteria
        if not (len(password1) >= 8 and any(c.isupper() for c in password1)
                and any(c.islower() for c in password1) and any(c.isdigit() for c in password1)
                and any(c in string.punctuation for c in password1)):
            return render_template('signup.html', message='Password criteria not met')

        hashed_password = bcrypt.hashpw(password1.encode('utf-8'), bcrypt.gensalt())

        # Check if email already exists
        existing_user = users_collection.find_one({'email': email})
        if existing_user:
            return render_template('signup.html', message='Email already exists. If you already have an account, please log in.')

        # Insert new user into users_collection
        user_data = {
            'name': name,
            'email': email,
            'mobile': mobile,
            'password': hashed_password
        }
        users_collection.insert_one(user_data)

        return redirect('/login')

    return render_template('signup.html')


@app.route('/logout')
def logout():
    # Remove user from session
    session.pop('user', None)
    
    return redirect('/')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Find user by email
        user = users_collection.find_one({'email': email})

        if user:
            # Convert ObjectId to string
            user['_id'] = str(user['_id'])

            # Check if password matches
            if bcrypt.checkpw(password.encode('utf-8'), user['password']):
                # Store user data in session
                session['user'] = user
                return redirect('/')
            else:
                return render_template('login.html', message='Incorrect password')
        else:
            return render_template('login.html', message='User not found')

    return render_template('login.html')

def login_required(func):
    @wraps(func)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    return decorated_function
@app.route('/disease')
@login_required
def result_page():
    return render_template('disease.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get input symptoms from request
    input_symptoms = request.json
    print(input_symptoms)
    # If no symptoms are selected, return an error response
    if not input_symptoms:
        return jsonify({'error': 'Please select at least one symptom.'}), 400

    # Update symptom values
    input_data = symptoms.copy()
    for symptom in input_symptoms:
        if symptom in input_data:
            input_data[symptom] = 1

    # Prepare test data
    df_test = pd.DataFrame(columns=list(input_data.keys()))
    df_test.loc[0] = np.array(list(input_data.values()))

    # Predict the disease
    predicted_disease = clf.predict(df_test)
    description = get_disease_description(predicted_disease)
    
    return jsonify({
        'predicted_disease': predicted_disease[0],
        'description': description
    })

@app.route('/hospital', methods=['GET', 'POST'])
@login_required
def hospital():
    return render_template('hospital.html')
@app.route('/dia')
@login_required
def dia():
    return render_template('diabetes.html')

@app.route('/pre', methods=['POST'])
def pre():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        
        # Predict the class
        my_prediction = classifier.predict(data)
        
        # Get the probability of having diabetes
        probabilities = classifier.predict_proba(data)
        diabetes_probability = probabilities[0][1] * 100  # Assuming class 1 is diabetes
        
        return render_template('diabetesresult.html', prediction=my_prediction, probability=diabetes_probability)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')

    text1 = "i need information related to disease"
    text2 = "please give me related to disease only"

    payload = {
        "providers": "openai",
        "texts": [text1, text2],
        "question": question,
        "examples_context": "In 2017, U.S. life expectancy was 78.6 years.",
        "examples": [["What is human life expectancy in the United States?", "78 years."]],
        "fallback_providers": ""
    }

    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        answer = result['openai']['answers'][0]
    else:
        answer = "Sorry, something went wrong."

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

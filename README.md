```markdown
# Disease Prediction Web Application

This Flask-based web application predicts diseases based on symptoms selected by the user. It integrates machine learning models for prediction and provides functionalities for user authentication, disease description, hospital location services, and diabetes prediction.

## Features

- **User Authentication**: Sign up, login, and logout using Flask sessions and MongoDB for user data storage.
- **Disease Prediction**: Predicts diseases based on selected symptoms using a pre-trained machine learning model.
- **Hospital Locator**: Retrieves major hospitals near a specified location using geolocation APIs.
- **Diabetes Prediction**: Provides prediction and probability of diabetes based on input parameters.
- **Chatbot Interface**: Allows users to ask questions related to diseases using OpenAI's language model.

## Technologies Used

- **Backend**: Flask, Python
- **Database**: MongoDB
- **Machine Learning**: Scikit-learn
- **Frontend**: HTML, Bootstrap, JavaScript
- **External APIs**: EdenAI, Overpass API

## Setup Instructions

1. **Install Dependencies**
   pip install -r requirements.txt


2. **Configure MongoDB**
   - Install MongoDB locally or set up a remote MongoDB instance.
   - Update the MongoDB URI in `app.py` to point to your database.

3. **Run the Application**
   python app.py
   The application will run locally at `http://localhost:5000`.

4. **Access the Application**
   Open a web browser and go to `http://localhost:5000` to access the web application.
```

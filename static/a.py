from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np


#initialize flask application
app = Flask(__name__)

#load pickle model and save it in a variable, model
model = pickle.load(open('model.pkl', 'rb'))

# Load the MinMaxScaler from the pickle file
scaler = pickle.load(open('min_max_scaler.pkl', 'rb'))



# route for the landing page
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():   
    
    # POST method is used to send html form data to the server
    if request.method == 'POST':
        # get all the form values and convert it to float, request.form.values() fetches the form values from the server
        int_features = [float(x) for x in request.form.values()]
        
      
        # The nimerical features that need scaling are: BMI (index 5), Sleep (index 8), SoundSleep (index 9), Pregnancies (index 14)
        numerical_indices = [5, 8, 9, 14]
        #numerical_features = [int_features[i] for i in numerical_indices]
        
        # Extract and scale the numerical features. reshape(1,-1) converts the 1D to 2D for the scaler. The flatten() function converts the scaled numerical values bacK to 1D so as to replace the scaled features with the original numerical features which is in 1D 
        numerical_features = np.array([int_features[i] for i in numerical_indices]).reshape(1, -1)
        numerical_features_scaled = scaler.transform(numerical_features).flatten()
        
        #print the scaled numerical features
        print('Scaled numerical features:', numerical_features_scaled)
        print("Scaler max:", scaler.data_max_)
        
        # Replace the original numerical features with their scaled versions
        for i, index in enumerate(numerical_indices):
            int_features[index] = numerical_features_scaled[i]
        
        # Convert the list to a numpy array and reshape for the model, in the form [[a,b,c,........]]
        features = np.array(int_features).reshape(1, -1)
        
        print('All features:', features)
        
        
        
        
        
         # Make prediction
        probabilities = model.predict_proba(features)
        
        # # Get the probability of the positive class (risk of developing diabetes)
        risk_score = probabilities[0][1] * 100  # Convert to percentage
        
        
        print('prob:', probabilities)
        print('score:', risk_score)

        # Return the result
        if risk_score >= 50:
            return render_template('result.html', prediction_text=f"You are at Risk of developing diabetes with a risk score of {risk_score:.2f}%", pt = "high risk")
        else:
            return render_template('result.html', prediction_text=f"You are not at Risk of developing diabetes with a risk score of {risk_score:.2f}%", pt = "low risk" )
        
        
        
        
    
        # # make prediction 
        # prediction = model.predict(features)
        # #get the prediction output
        # output = prediction[0]
    
        # if output == 1:
        #     return render_template('result.html', prediction_text = "You are at Risk of developing diabetes")
        # else:
        #     return render_template('result.html', prediction_text = "You are not at Risk")
    
        
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run()
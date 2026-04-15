from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('../fetal_health1.pkl', 'rb'))

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Retrieve values from UI
            prolongued_decelerations = float(request.form['prolongued_decelerations'])
            abnormal_short_term_variability = float(request.form['abnormal_short_term_variability'])
            percentage_of_time_with_abnormal_long_term_variability = float(request.form['percentage_of_time_with_abnormal_long_term_variability'])
            histogram_variance = float(request.form['histogram_variance'])
            histogram_median = float(request.form['histogram_median'])
            mean_value_of_long_term_variability = float(request.form['mean_value_of_long_term_variability'])
            histogram_mode = float(request.form['histogram_mode'])
            accelerations = float(request.form['accelerations'])

            # Create the array for prediction matching the exact training feature order
            X = [[
                accelerations, 
                prolongued_decelerations, 
                abnormal_short_term_variability, 
                percentage_of_time_with_abnormal_long_term_variability,
                mean_value_of_long_term_variability,
                histogram_variance,
                histogram_median,
                histogram_mode
            ]]

            # Predict
            output = model.predict(X)
            
            # Map dataset classifications (1=Normal, 2=Suspect, 3=Pathological)
            pred_value = int(output[0])
            if pred_value == 1:
                result = 'Normal'
            elif pred_value == 2:
                result = 'Suspect'
            else:
                result = 'Pathological'

            return render_template('output.html', output=result)
        
        except Exception as e:
            return f"An error occurred: {e}"

    # If it's a GET request, render the input form
    return render_template("inspect.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
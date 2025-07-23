from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        gender = request.form['gender']
        married = request.form['married']
        education = request.form['education']
        self_employed = request.form['self_employed']
        applicant_income = float(request.form['applicant_income'])
        loan_amount = float(request.form['loan_amount'])

        # Convert categorical to numeric
        gender = 1 if gender == 'Male' else 0
        married = 1 if married == 'Yes' else 0
        education = 1 if education == 'Graduate' else 0
        self_employed = 1 if self_employed == 'Yes' else 0

        # Final input
        input_data = np.array([[gender, married, education, self_employed, applicant_income, loan_amount]])

        prediction = model.predict(input_data)[0]
        result = "Loan Approved ✅" if prediction == 1 else "Loan Rejected ❌"

        return render_template('prediction.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)

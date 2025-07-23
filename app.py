from flask import Flask, render_template, request
import numpy as np
import pickle
import math

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Raw Inputs
            gender = request.form["Gender"]
            married = request.form["Married"]
            dependents = request.form["Dependents"]
            education = request.form["Education"]
            self_employed = request.form["Self_Employed"]
            credit_history = float(request.form["Credit_History"])
            property_area = request.form["Property_Area"]
            income = float(request.form["ApplicantIncome"])
            loan_amount = float(request.form["LoanAmount"])

            # Feature engineering
            applicantincomelog = np.log(income + 1)
            loan_amount_log = np.log(loan_amount + 1)

            # Manual One-Hot Encoding
            Gender_Male = 1 if gender == "Male" else 0
            Married_Yes = 1 if married == "Yes" else 0
            Dependents_1 = 1 if dependents == "1" else 0
            Dependents_2 = 1 if dependents == "2" else 0
            Dependents_3plus = 1 if dependents == "3+" else 0
            Education_NotGraduate = 1 if education == "Not Graduate" else 0
            Self_Employed_Yes = 1 if self_employed == "Yes" else 0
            Property_Semiurban = 1 if property_area == "Semiurban" else 0
            Property_Urban = 1 if property_area == "Urban" else 0

            # Final feature array
            features = [[
                credit_history,
                applicantincomelog,
                loan_amount_log,
                Gender_Male,
                Married_Yes,
                Dependents_1,
                Dependents_2,
                Dependents_3plus,
                Education_NotGraduate,
                Self_Employed_Yes,
                Property_Semiurban,
                Property_Urban
            ]]

            prediction = model.predict(features)

            result = "Approved" if prediction[0] == 1 else "Rejected"

            if result == "Approved":
                interest = round(loan_amount * 0.10, 2)
                total = round(loan_amount + interest, 2)
            else:
                interest = total = 0

            return render_template("prediction.html", result=result, loan_amount=loan_amount, interest=interest, total=total)

        except Exception as e:
            return render_template("prediction.html", result="Error: " + str(e))

    return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)

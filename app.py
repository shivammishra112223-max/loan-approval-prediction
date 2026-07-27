import joblib
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

model=joblib.load("loan_model.pkl")
scaler=joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    no_of_dependents = int(request.form["no_of_dependents"])
    education = int(request.form["education"])
    self_employed = int(request.form["self_employed"])
    income_annum = float(request.form["income_annum"])
    loan_amount = float(request.form["loan_amount"])
    loan_term = float(request.form["loan_term"])
    cibil_score = float(request.form["cibil_score"])
    residential_assets_value = float(request.form["residential_assets_value"])
    commercial_assets_value = float(request.form["commercial_assets_value"])
    luxury_assets_value = float(request.form["luxury_assets_value"])
    bank_asset_value = float(request.form["bank_asset_value"])
    
    new_customer = pd.DataFrame({

    "no_of_dependents":[no_of_dependents],
    "education":[education],
    "self_employed":[self_employed],
    "income_annum":[income_annum],
    "loan_amount":[loan_amount],
    "loan_term":[loan_term],
    "cibil_score":[cibil_score],
    "residential_assets_value":[residential_assets_value],
    "commercial_assets_value":[commercial_assets_value],
    "luxury_assets_value":[luxury_assets_value],
    "bank_asset_value":[bank_asset_value]

})
    
    
    scale_columns = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

    new_customer[scale_columns] = scaler.transform(
    new_customer[scale_columns]
)



    prediction = model.predict(new_customer)

    probability = model.predict_proba(new_customer)

    approve_probability = round(probability[0][1] * 100, 2)
    
    if approve_probability >= 80:
        confidence = "High"
    
    elif approve_probability >= 60:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"
    
    print("Prediction:", result)
    print("Probability:", approve_probability)
    print("Confidence:", confidence)
    
    
    
    return render_template(

    "index.html",

    prediction=result,
    probability=approve_probability,
    confidence= confidence


)        
    
    
if __name__ == "__main__":
     app.run(debug=True)
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, flash, redirect, url_for , session
from forms import user_details
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


app.secret_key='123456789'
@app.route('/')
@app.route('/page', methods=['GET', 'POST'])
def details():
    formdata = user_details();
    if request.method == 'POST':
        if float(request.form['loan_amount_term'])<=0 or float(request.form['loan_amount'])<=0 or float(request.form['Coapplicant_income'])<=0 or float(request.form['applicant_income'])<=0:
            nagdata="Negative values not allowed"
            session['nag']=nagdata
          
            app.secret_key = '8866104644'
            return render_template('test.html', title='Details', form=formdata)
        else:
            applicant_name = request.form['applicant_name']
            Gender = str(request.form['gender'])
            Married = str(request.form['married'])
            Dependents = str(request.form['dependents'])
            Education = str(request.form['education'])
            Self_Employed = str(request.form['self_employed'])
            Credit_History = str(request.form['credit_history'])
            Property_Area = str(request.form['property_area'])
            ApplicantIncome = float(request.form['applicant_income'])
            CopplicantIncome = float(request.form['Coapplicant_income'])
            Loan_Amount = float(request.form['loan_amount'])
            Loan_Amount_Term = float(request.form['loan_amount_term'])     
            res_df = pd.DataFrame()
            res_df["Loan_ID"] = [applicant_name]
            res_df["Gender"] = [Gender]
            res_df["Married"] = [Married]
            res_df["Dependents"] = [Dependents]
            res_df["Education"] = [Education]
            res_df["Self_Employed"] = [Self_Employed]
            res_df["ApplicantIncome"] = [ApplicantIncome]
            res_df["CoapplicantIncome"] = [CopplicantIncome]
            res_df["LoanAmount"] = [Loan_Amount]
            res_df["Loan_Amount_Term"] = [Loan_Amount_Term]
            res_df["Credit_History"] = [Credit_History]
            res_df["Property_Area"] = [Property_Area]
            user_data = res_df
            test_sample = pd.read_csv(r'test_sample.csv')
            test_sample = test_sample.iloc[:,1:]
            df = test_sample.append(user_data)
            df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
            df = df.drop(["ApplicantIncome", "CoapplicantIncome"], axis=1)
            r = 7
            p = df["LoanAmount"]*1000
            n = df["Loan_Amount_Term"]
            emi = (p * r * (1 + r) * n) / ((1 + r) * n - 1)
            df["EMI"] = pd.Series(emi)
            df = df.drop("Loan_Amount_Term", axis=1)
            df["Capacity"] = (1 - (df.LoanAmount / df.TotalIncome)) * 100
            clean_data = df.drop("Loan_ID", axis=1)
            categorical = [var for var in clean_data.columns if clean_data[var].dtype == 'O']
            categorical.remove('Credit_History')
            cat_clean_data = clean_data[categorical]
            final_data = pd.get_dummies(cat_clean_data)
            final_data[["LoanAmount", "TotalIncome", "EMI", "Capacity"]] = clean_data[["LoanAmount", "TotalIncome", "EMI", "Capacity"]]
            ch = pd.DataFrame(pd.get_dummies(clean_data["Credit_History"]))
            final_data[ch.columns] = ch[ch.columns]
            final_data = final_data.rename({0: 'Credit_History_0', 1: "Credit_History_1"}, axis=1)
            col = [a for a in range(final_data.shape[1])]
            col.remove(4)
            final_data = final_data.iloc[:, col]
            final_data["LoanAmount_log"] = np.log(final_data.LoanAmount)
            final_data = final_data.drop("LoanAmount", axis=1)
            final_data["TotalIncome_log"] = np.log(final_data.TotalIncome)
            final_data = final_data.drop("TotalIncome", axis=1)
            final_data["EMI_log"] = np.log(final_data.EMI)
            final_data = final_data.drop("EMI", axis=1)
            predict_user_data = final_data.tail(1)
            user_emi = ((np.exp(predict_user_data["EMI_log"]))/100)*2
            
            pred = model.predict(predict_user_data)
            pred_proba = model.predict_proba(predict_user_data)
            if pred[0] == 'Y':
                pred = 'YES'
            else:
                pred = 'NO'
            
            n_prob = pred_proba[0][0] * 100
            y_prob = pred_proba[0][1] * 100
            
            session['y_prob']=round(y_prob,2)
            session['n_prob']=round(n_prob,2)
            session['user_emi']=user_emi[0]
            session['pred']=pred

                   
            return redirect(url_for('details'))

    app.secret_key = '6355320573'
    return render_template('detail.html', title='Details', form=formdata)

if __name__ == '__main__':
    app.run(debug=True)


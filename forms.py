from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, SelectField, DecimalField
from wtforms.validators import DataRequired, Length, Email, EqualTo, NumberRange
from flask import request

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, IntegerField
from wtforms.validators import DataRequired, Length, Email, EqualTo, NumberRange, required
from wtforms import validators, widgets
from wtforms.fields import html5 as h5fields
from wtforms.widgets import html5 as h5widgets
from wtforms.validators import number_range


class user_details(FlaskForm):
    applicant_name = StringField('Applicant_Name',
                                 validators=[DataRequired(), Length(min=2, max=15)])
    gender = SelectField('Gender', choices=[('Male', 'Male'), ('Female', 'Female')], validators=[DataRequired()])
    married = SelectField('Married', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    dependents = SelectField('Dependents', choices=[('0', '0'), ('1', '1'), ('2', '2'), ('3+', '3+')], validators=[DataRequired()])
    education = SelectField('Education', choices=[('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')], validators=[DataRequired()])
    self_employed = SelectField('Self_Employed', choices=[('Yes', 'Yes'), ('No', 'No')], validators=[DataRequired()])
    credit_history = SelectField('Credit_History', choices=[('1', 'Yes'), ('0', 'No')], validators=[DataRequired()])
    property_area = SelectField('Property_Area', choices=[('Urban', 'Urban'), ('Rural', 'Rural'), ('Semiurban', 'Semiurban')], validators=[DataRequired()])
    applicant_income = h5fields.DecimalField('Applicant_Income', validators=[required(), NumberRange(min=0, max=100000)])
    Coapplicant_income = h5fields.DecimalField('Coapplicant_Income', validators=[required(), NumberRange(min=0, max=100000)])
    loan_amount = h5fields.DecimalField('Loan_Amount', validators=[required(), NumberRange(min=0, max=100000)],
                                        render_kw={'placeholder':'Loan amount in thousands'})
    loan_amount_term = h5fields.IntegerField('Loan_Amount_Term',  validators=[required(), number_range(min=-10000000000000, max=0, message="positive number only")],
                                             render_kw={'placeholder':'Numbers of month to repay loan'})
    submit = SubmitField('Predict')


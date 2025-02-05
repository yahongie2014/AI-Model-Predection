import datetime
import difflib
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, abort, make_response
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle
from validator import RsForm, ScoreForm, ProfitabilityForm, FeedForm, DelayForm, PayoutForm
from datetime import datetime as dtime

app = Flask(__name__)

# Linux Path
# -------------------------------------------------------------------------
main_dict_path = '//var//www/html//models//dictionaries_main.sav'
prof_dict_path = '//var//www/html//models//dictionaries_main.sav'
feed_dict_path = '//var//www/html//models//dictionaries_main.sav'
delay_dict_path = '//var//www/html//models//dictionaries_main.sav'
path = '//var//www/html//models//trainIncorta.pkl'
code_path = '//var//www/html//models//scoree.sav'
rf_model_prof_path = '//var//www/html//models//rf_model_fitted_profitability.sav'
rf_model_fitted_Feedback_path = '//var//www/html//models//rf_model_fitted_Feedback.sav'
rf_model_fitted_Delay_path = '//var//www/html//models//rf_model_fitted_Delay.sav'
RF_Regressor_path = '//var//www/html//models//RF_Regressor.sav'
csv_path = '//var//www/html//models//RsIncorta.csv'
csv_path_files = 'var\\www\\html\\csv'
json_file_path= '//var//www/html//main_dictionary.json'
output_file_path = '//var//www/html//models//dictionaries_main.sav'
# -------------------------------------------------------------------------



# Mac Path
# -------------------------------------------------------------------------
# main_dict_path = 'models/dictionaries_main.sav'
# prof_dict_path = 'models/dictionaries_main.sav'
# feed_dict_path = 'models/dictionaries_main.sav'
# delay_dict_path = 'models/dictionaries_main.sav'
# path = 'models/trainIncorta.pkl'
# code_path = 'models/scoree.sav'
# rf_model_prof_path = 'models/rf_model_fitted_profitability.sav'
# rf_model_fitted_Feedback_path = 'models/rf_model_fitted_Feedback.sav'
# rf_model_fitted_Delay_path = 'models/rf_model_fitted_Delay.sav'
# RF_Regressor_path = 'models/RF_Regressor.sav'
# csv_path = 'models//RsIncorta.csv'
# json_file_path= 'GenerateModel/main_dictionary.json'
# output_file_path = 'dictionaries_main.sav'
# -------------------------------------------------------------------------

# Windows Path
# -------------------------------------------------------------------------
# original_path = 'D:\Python Script\AI PythonScipt'
# csv_path = os.path.join(original_path, 'models\RsIncorta.csv')
# path = os.path.join(original_path, 'models\\trainIncorta.pkl')
# code_path = os.path.join(original_path, 'models\scoree.sav')
# prof_dict = os.path.join(original_path, 'models\dictionaries_profitability.sav')
# feed_dict = os.path.join(original_path, 'models\dictionaries_feedback.sav')
# delay_dict = os.path.join(original_path, 'models\dictionaries_delay.sav')
# rf_model_prof = os.path.join(original_path, 'models\\rf_model_fitted_profitability.sav')
# rf_model_fitted_Feedback = os.path.join(original_path, 'models\\rf_model_fitted_Feedback.sav')
# rf_model_fitted_Delay = os.path.join(original_path, 'models\\rf_model_fitted_Delay.sav')
# RF_Regressor = os.path.join(original_path, 'models\\RF_Regressor.sav')
# -------------------------------------------------------------------------

def define_dictionary(value, dictionary):
    lower_case_value = value.lower()
    lower_case_dict = {k.lower(): v for k, v in dictionary.items()}

    if lower_case_value in lower_case_dict:
        return lower_case_dict[lower_case_value]
    else:
        suggestions = difflib.get_close_matches(lower_case_value, lower_case_dict.keys())
        if suggestions:
            suggestion_str = ', '.join(suggestions)
            return f"Error: The key '{value}' was not found in the dictionary. Did you mean: {suggestion_str}?"
        else:
            return f"Error: The key '{value}' was not found in the dictionary and no similar keys were found."


def define_dictonary_value(key, dictionary):
    return dictionary.get(key)


# Return View With Parameters
@app.route('/')
def home():
    today = datetime.date.today()

    year = today.year

    return render_template('index.html', year=year)


# After Train And Post Form
@app.route('/predict', methods=['POST'])
def predict():
    with open(path, 'rb') as file:
        pickeld_model = joblib.load(file)

    # ['Task_Type', 'Brand', 'Account', 'Subject', 'Unit', 'Language_Pair', 'Tool', 'Rs_Plan',
    #          'unified_task_amount', 'DateStamp_base_2022', 'Duration', 'RS_Month', 'RS_M_Day', 'Hour']
    le = preprocessing.LabelEncoder()

    Task_Type = request.form.get("Task_Type")
    Brand = request.form.get("Brand")
    Subject = request.form.get('Subject')
    Language_Pair = request.form.get('Language_Pair')
    Unit = request.form.get('Unit')
    Tool = request.form.get('Tool')
    Rs_Plan = request.form.get('Rs_Plan')
    unified_task_amount = request.form.get("unified_task_amount")
    dateStamp_base = request.form.get('dateStamp_base')
    duration = request.form.get('duration')
    rs_month = request.form.get('rs_month')
    rs_m_day = request.form.get('rs_m_day')
    hour = request.form.get('hour')

    X_COLUMS = ['Task_Type', 'Brand', 'Subject', 'Unit', 'Language_Pair', 'Tool', 'Rs_Plan',
                'unified_task_amount', 'DateStamp_base_2022', 'Duration', 'RS_Month', 'RS_M_Day', 'Hour']
    X_test = np.array([[le.fit_transform([Task_Type]).shape[0], le.fit_transform([Brand]).shape[0],
                        le.fit_transform([Subject]).shape[0],
                        le.fit_transform([Unit]).shape[0], le.fit_transform([Language_Pair]).shape[0],
                        le.fit_transform([Tool]).shape[0], le.fit_transform([Rs_Plan]).shape[0],
                        unified_task_amount, dateStamp_base, duration, rs_month, rs_m_day, hour]])

    y_pred = pickeld_model.predict(X_test)
    print("Prediction>>>>>>", y_pred)

    score = pickeld_model.score(X_test, y_pred)

    # PERCENTAGE DATA SUCCESS
    percentage = pickeld_model.predict_proba(X_test[:1])
    print("Percentage>>>>>>", percentage)

    # IMPORTANCE COLUMN
    Importance = pickeld_model.feature_importances_

    columns = X_COLUMS[0]

    if y_pred == 1:
        result = 'Success'
    else:
        result = 'Failed'

    return render_template('index.html',
                           prediction_text='Task will be: ' + result + ' for brand ' +
                                           Brand + '  Fail Percentage : ' + (
                                                   f"{round(percentage[0][0] * 100, 2)}%" + '  Success Percentage: ' + (
                                               f"{round(percentage[0][1] * 100, 2)}%")))


# Train Model And Generate PKLE
@app.route('/train', methods=['GET'])
def train():
    df_raw = pd.read_csv(csv_path, sep=',',
                         encoding='unicode_escape',
                         low_memory=False)
    df_raw.head(10)
    df_raw.columns = df_raw.columns.str.replace(' ', '_')
    df_raw.info()
    sum(df_raw.duplicated())
    df_raw.isnull().all()
    df_raw.isnull().sum()
    df1 = df_raw.dropna(subset=['Task_Type', 'Brand', 'Subject', 'Unit', 'Language_Pair', 'Tool', 'Rs_Plan',
                                'unified_task_amount', 'DateStamp_base_2022', 'Duration', 'RS_Month', 'RS_M_Day',
                                'Hour'], how='any')
    df1

    label_encoder = preprocessing.LabelEncoder()
    df1["Task_Type"] = label_encoder.fit_transform(df1["Task_Type"])
    df1["Brand"] = label_encoder.fit_transform(df1["Brand"])
    df1["Subject"] = label_encoder.fit_transform(df1["Subject"])
    df1["Unit"] = label_encoder.fit_transform(df1["Unit"])
    df1["Language_Pair"] = label_encoder.fit_transform(df1["Language_Pair"])
    df1["Tool"] = label_encoder.fit_transform(df1["Tool"])
    df1["Rs_Plan"] = label_encoder.fit_transform(df1["Rs_Plan"])

    df1["Task_Type"] = pd.Categorical(df1["Task_Type"])
    df1["Brand"] = pd.Categorical(df1["Brand"])
    df1["Subject"] = pd.Categorical(df1["Subject"])
    df1["Unit"] = pd.Categorical(df1["Unit"])
    df1["Language_Pair"] = pd.Categorical(df1["Language_Pair"])
    df1["Tool"] = pd.Categorical(df1["Tool"])
    df1["Rs_Plan"] = pd.Categorical(df1["Rs_Plan"])
    X = df1[[
        'Task_Type', 'Brand', 'Subject', 'Unit', 'Language_Pair', 'Tool', 'Rs_Plan',
        'unified_task_amount', 'DateStamp_base_2022', 'Duration', 'RS_Month', 'RS_M_Day', 'Hour']]
    y = np.array(df1["RS_Success"]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)
    rf_model = RandomForestClassifier(n_estimators=50, max_features="auto", random_state=44)
    Fitter = rf_model.fit(X_train, y_train)
    with open(path, 'wb') as file:
        joblib.dump(Fitter, file)

    y_pred = rf_model.predict(X_test)
    meanAbsolute = mean_absolute_error(y_test, y_pred)
    meanSquared = mean_squared_error(y_test, y_pred)
    meanRoot = np.sqrt(mean_squared_error(y_test, y_pred))
    r2Score = r2_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred) * 100

    stastic = [
        {
            'Mean Absolute Error': meanAbsolute,
            'Mean Squared Error': meanSquared,
            'Root Mean Squared Error': r2Score,
            'R2 Score': meanRoot,
            'Accuracy': acc
        }
    ]
    return jsonify(stastic)


# After Train And Post by Api Request
@app.route('/api/predict', methods=['POST', 'GET'])
def predict_api():
    with open(path, 'rb') as file:
        pickeld_model = joblib.load(file)

    # ['Task_Type', 'Brand', 'Account', 'Subject', 'Unit', 'Language_Pair', 'Tool', 'Rs_Plan',
    #          'unified_task_amount', 'DateStamp_base_2022', 'Duration', 'RS_Month', 'RS_M_Day', 'Hour']

    request_data = request.form
    le = preprocessing.LabelEncoder()

    unified_task_amount = request_data["unified_task_amount"]
    dateStamp_base = request_data['dateStamp_base']
    duration = request_data['duration']
    rs_month = request_data['rs_month']
    rs_m_day = request_data['rs_m_day']
    hour = request_data['hour']
    Task_Type = request_data["Task_Type"]
    Brand = request_data["Brand"]
    Subject = request_data["Subject"]
    Language_Pair = request_data["Language_Pair"]
    Unit = request_data["Unit"]
    Rs_Plan = request_data["Rs_Plan"]
    Tool = request_data["Tool"]

    form = RsForm(request.form)

    if form.validate():
        X_COLUMS = ['Task_Type', 'Brand', 'Subject', 'Unit', 'Language_Pair', 'Tool', 'Rs_Plan',
                    'unified_task_amount', 'DateStamp_base_2022', 'Duration', 'RS_Month', 'RS_M_Day', 'Hour']
        X_test = np.array([[le.fit_transform([Task_Type]).shape[0], le.fit_transform([Brand]).shape[0],
                            le.fit_transform([Subject]).shape[0],
                            le.fit_transform([Unit]).shape[0], le.fit_transform([Language_Pair]).shape[0],
                            le.fit_transform([Tool]).shape[0], le.fit_transform([Rs_Plan]).shape[0],
                            unified_task_amount, dateStamp_base, duration, rs_month, rs_m_day, hour]])

        # PREDICTION OF SUCCESS OR FAIL
        y_pred = pickeld_model.predict(X_test)
        print("Prediction>>>>>>", y_pred)

        score = pickeld_model.score(X_test, y_pred)

        # PERCENTAGE DATA SUCCESS
        percentage = pickeld_model.predict_proba(X_test[:1])
        print("Percentage>>>>>>", percentage)

        # IMPORTANCE COLUMN
        Importance = pickeld_model.feature_importances_

        columns = X_COLUMS[0]
        i = 0
        while i < len(columns):
            print(f" the importance of feature '{columns[i]}' is {round(Importance[i] * 100, 2)}%.")
            i += 1

        if y_pred == 1:
            Pred = (f"Task For Brand {Brand} Will Be Success")
        else:
            Pred = (f"Task For Brand {Brand} Will Be Failed")
        stastic = {
            "Prediction": Pred,
            "Fail_Percentage": (f"{round(percentage[0][0] * 100, 2)}"),
            "Success_Percentage": (f"{round(percentage[0][1] * 100, 2)}"),
            "Prediction_Score": int(y_pred)
        }
        return jsonify(success=True, data=stastic)
    else:
        return jsonify(success=False, errors=form.errors)


@app.route('/api/checkScore', methods=['POST', 'GET'])
def checkScore():
    pickeld_model = pickle.load(open(code_path, 'rb'))
    request_data = request.form
    Vendor_types_name = request_data["Vendor_types_name"]
    total_gross_amount_Main_currency = request_data['total_gross_amount_Main_currency']
    Count_Tasks = request_data['Count_Tasks']
    diff_Due_task = request_data['diff_Due_task']
    cost_day = request_data['cost_day']
    task_amount_days = request_data['task_amount_days']

    form = ScoreForm(request_data)

    if form.validate():
        if (Vendor_types_name == 'Freelancer'):
            code = 1
        else:
            code = 0

        anomaly_inputs = np.array(
            [[code, total_gross_amount_Main_currency, Count_Tasks,
              diff_Due_task, task_amount_days,
              cost_day]], dtype=float)
        anomaly = pickeld_model.predict(anomaly_inputs)
        anomaly_scores = pickeld_model.decision_function(anomaly_inputs)
        fraud = np.where((anomaly < 0), 1, 0)
        stastic = {
            "score": anomaly_scores.tolist()[0],
            "fraud": fraud.tolist()[0],

        }
        return jsonify(success=True, data=stastic)
    else:
        return jsonify(success=False, errors=form.errors)


@app.route('/api/profitability', methods=['POST', 'GET'])
def profitability():
    prof_dictionary = pickle.load(open(prof_dict_path, 'rb'))

    PMS = prof_dictionary['PM']
    Brands = prof_dictionary['Brand']
    Accounts = prof_dictionary['Account']
    Job_types = prof_dictionary['Job_type']
    Language_Pairs = prof_dictionary['Language_Pair']
    Subjects = prof_dictionary['Subject']
    Units = prof_dictionary['Unit']
    profitability = prof_dictionary['profitability']


    try:
        with open(rf_model_prof_path, 'rb') as model_file:
            prof_model = pickle.load(model_file)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"File not found: {rf_model_prof_path}")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")

    request_data = request.form
    Brand = request_data["Brand"]
    Unit = request_data['Unit']
    Job_type = request_data['Job_type']
    Subject = request_data['Subject']
    Language_Pair = request_data['Language_Pair']
    Start_TimeStamp = dtime.strptime(request_data['Start_TimeStamp'], '%Y-%m-%d %H:%M:%S')
    Deivery_TimeStamp = dtime.strptime(request_data['Deivery_TimeStamp'], '%Y-%m-%d %H:%M:%S')
    Price = request_data['Price']
    amount = request_data['amount']
    PM = request_data['PM']
    Account = request_data['Account']

    base_date = dtime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    difference_start = Start_TimeStamp - base_date
    difference_delivery = Deivery_TimeStamp - base_date
    total_days_start = difference_start.days
    fractional_days_start = difference_start.seconds / (24 * 3600)

    total_days_delivery = difference_delivery.days
    fractional_days_delivery = difference_delivery.seconds / (24 * 3600)

    total_days_with_fraction_start = total_days_start + fractional_days_start
    total_days_with_fraction_delivery = total_days_delivery + fractional_days_delivery
    duration_final = total_days_with_fraction_delivery - total_days_with_fraction_start

    form = ProfitabilityForm(request_data)
    if form.validate():
        X_COLUMS = ['Brand', 'Unit', 'Job_type', 'Subject', 'Language_Pair', 'Start_TimeStamp',
                    'Price', 'Deivery_TimeStamp', 'amount', 'Duration', 'PM', 'Account']
        Requests = [define_dictionary(Brand, Brands), define_dictionary(Unit, Units), define_dictionary(Job_type, Job_types),
              define_dictionary(Subject, Subjects), define_dictionary(Language_Pair, Language_Pairs), (f'{total_days_with_fraction_start:.2f}'),
              Price, (f'{total_days_with_fraction_delivery:.2f}'), amount, duration_final, define_dictionary(PM, PMS),
              define_dictionary(Account, Accounts)]


        for req in Requests:
            if isinstance (req, str) and req.startswith ("Error"):
                return make_response(jsonify(success=False, data=req),500)

        Requests = np.array ([Requests])

        predection = prof_model.predict(Requests)
        prob = prof_model.predict_proba(Requests)
        dfn1 = pd.DataFrame(prob, columns=["Low", "Normal", "High"])

        dfn1['y_pred'] = predection
        dfn1['25_pred'] = np.where((dfn1['Low'] >= 0.25), 0, dfn1['y_pred'])

        if (dfn1['25_pred'][0] == 0):
            percentage = dfn1['Low'][0]
        elif (dfn1['25_pred'][0] == 1):
            percentage = dfn1['Normal'][0]
        elif (dfn1['25_pred'][0] == 2):
            percentage = dfn1['High'][0]

        if (dfn1['25_pred'][0] == profitability['Low']):
            profitability_var = 'Low'
        elif (dfn1['25_pred'][0] == profitability['Normal']):
            profitability_var = 'Normal'
        elif (dfn1['25_pred'][0] == profitability['High']):
            profitability_var = 'High'

        stastic = {
            "profitability": profitability_var,
            "25_pred": dfn1['25_pred'].tolist()[0],
            "percentage": (f"{round(percentage * 100, 2)}%"),
        }
        return jsonify(success=True, data=stastic)
    else:
        return jsonify(success=False, errors=form.errors)


@app.route('/api/feedback', methods=['POST', 'GET'])
def feedback():
    feed_dictionary = pickle.load(open(feed_dict_path, 'rb'))
    feed_model = pickle.load(open(rf_model_fitted_Feedback_path, 'rb'))
    PMS = feed_dictionary['PM']
    Brands = feed_dictionary['Brand']
    Accounts = feed_dictionary['Account']
    Job_types = feed_dictionary['Job_type']
    Language_Pairs = feed_dictionary['Language_Pair']
    Subjects = feed_dictionary['Subject']
    Units = feed_dictionary['Unit']

    request_data = request.form
    Brand = request_data["Brand"]
    Unit = request_data['Unit']
    Job_type = request_data['Job_type']
    Subject = request_data['Subject']
    Language_Pair = request_data['Language_Pair']
    Start_TimeStamp = dtime.strptime(request_data['Start_TimeStamp'], '%Y-%m-%d %H:%M:%S')
    Deivery_TimeStamp = dtime.strptime(request_data['Deivery_TimeStamp'], '%Y-%m-%d %H:%M:%S')
    Price = request_data['Price']
    amount = request_data['amount']
    PM = request_data['PM']
    Account = request_data['Account']

    form = FeedForm(request_data)

    base_date = dtime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    difference_start = Start_TimeStamp - base_date
    difference_delivery = Deivery_TimeStamp - base_date
    total_days_start = difference_start.days
    fractional_days_start = difference_start.seconds / (24 * 3600)

    total_days_delivery = difference_delivery.days
    fractional_days_delivery = difference_delivery.seconds / (24 * 3600)

    total_days_with_fraction_start = total_days_start + fractional_days_start
    total_days_with_fraction_delivery = total_days_delivery + fractional_days_delivery
    duration_final = total_days_with_fraction_delivery - total_days_with_fraction_start

    if form.validate():
        X_COLUMS = ['Brand', 'Unit', 'Job_type', 'Subject', 'Language_Pair', 'Start_TimeStamp',
                    'Price', 'Deivery_TimeStamp', 'amount', 'Duration', 'PM', 'Account']
        Requests = [define_dictionary(Brand, Brands), define_dictionary(Unit, Units), define_dictionary(Job_type, Job_types),
              define_dictionary(Subject, Subjects),define_dictionary(Language_Pair, Language_Pairs), total_days_with_fraction_start,
              Price, total_days_with_fraction_delivery, amount, duration_final, define_dictionary(PM, PMS),
              define_dictionary(Account, Accounts)]

        for req in Requests:
            if isinstance (req, str) and req.startswith ("Error"):
                return make_response(jsonify(success=False, data=req),500)

        Requests = np.array([Requests])

        predection = feed_model.predict(Requests)
        prob = feed_model.predict_proba(Requests)



        dfn1 = pd.DataFrame(prob, columns=['positive', 'negative'])
        dfn1['y_pred'] = predection
        dfn1['9_pred'] = np.where((dfn1['negative'] > 0.09), 1, 0)
        if (dfn1['9_pred'][0] == 0):
            stastic = {
                "Percentage": (f"{round(dfn1['positive'].tolist()[0] * 100, 2)}%"),
                "9_pred": dfn1['9_pred'].tolist()[0],
                "status": "Positive",
            }
        else:
            stastic = {
                "Percentage": (f"{round(dfn1['negative'].tolist()[0] * 100, 2)}%"),
                "9_pred": dfn1['9_pred'].tolist()[0],
                "status": "Negative",
            }

        return jsonify(success=True, data=stastic)
    else:
        return jsonify(success=False, errors=form.errors)


@app.route('/api/Delay', methods=['POST', 'GET'])
def Delay():
    delay_dictionary = pickle.load(open(delay_dict_path, 'rb'))
    delay_model = pickle.load(open(rf_model_fitted_Delay_path, 'rb'))

    PMS = delay_dictionary['PM']
    Brands = delay_dictionary['Brand']
    Accounts = delay_dictionary['Account']
    Job_types = delay_dictionary['Job_type']
    Language_Pairs = delay_dictionary['Language_Pair']
    Subjects = delay_dictionary['Subject']
    Units = delay_dictionary['Unit']
    Delays = delay_dictionary['Delay']

    request_data = request.form
    Brand = request_data["Brand"]
    Unit = request_data['Unit']
    Job_type = request_data['Job_type']
    Subject = request_data['Subject']
    Language_Pair = request_data['Language_Pair']
    Start_TimeStamp = dtime.strptime(request_data['Start_TimeStamp'], '%Y-%m-%d %H:%M:%S')
    Deivery_TimeStamp = dtime.strptime(request_data['Deivery_TimeStamp'], '%Y-%m-%d %H:%M:%S')
    amount = request_data['amount']
    PM = request_data['PM']
    Account = request_data['Account']

    form = DelayForm(request_data)

    base_date = dtime.strptime('2020-01-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    difference_start = Start_TimeStamp - base_date
    difference_delivery = Deivery_TimeStamp - base_date
    total_days_start = difference_start.days
    fractional_days_start = difference_start.seconds / (24 * 3600)

    total_days_delivery = difference_delivery.days
    fractional_days_delivery = difference_delivery.seconds / (24 * 3600)

    total_days_with_fraction_start = total_days_start + fractional_days_start
    total_days_with_fraction_delivery = total_days_delivery + fractional_days_delivery
    duration_final = total_days_with_fraction_delivery - total_days_with_fraction_start

    if form.validate():
        X_COLUMS = ['Brand', 'Unit', 'Job_type', 'Subject', 'Language_Pair', 'Start_TimeStamp',
                    'Deivery_TimeStamp', 'amount', 'Duration', 'PM', 'Account']
        Requests = [
            define_dictionary(Brand, Brands),
            define_dictionary(Unit, Units),
            define_dictionary(Job_type, Job_types),
            define_dictionary(Subject, Subjects),
            define_dictionary(Language_Pair, Language_Pairs),
            total_days_with_fraction_start,
            total_days_with_fraction_delivery,
            amount,
            duration_final,
            define_dictionary(PM, PMS),
            define_dictionary(Account, Accounts)
        ]

        for req in Requests:
            if isinstance(req, str) and req.startswith("Error"):
                return make_response(jsonify(success=False, data=req),500)

        Requests = np.array([Requests])
        y_pred = delay_model.predict(Requests)
        y_test = delay_model.predict(Requests)
        prob = delay_model.predict_proba(Requests)

        dfn1 = pd.DataFrame(prob, columns=["On Time", "Delayed"])
        dfn1['y_pred'] = y_pred
        dfn1['y_test'] = y_test
        dfn1['32_pred'] = np.where((dfn1['Delayed'] > 0.32), 1, 0)


        if(dfn1['32_pred'].tolist()[0] == 1):
            delaStatus = 'Delayed'
        else:
            delaStatus = 'On Time'

        if (dfn1['32_pred'][0] == 1):
            stastic = {
                "32_pred": dfn1['32_pred'].tolist()[0],
                "Percentage": (f"{round(dfn1['Delayed'].tolist()[0] * 100, 2)}%"),
                "status": 'Delayed',
            }
        else:
            stastic = {
                "32_pred": dfn1['32_pred'].tolist()[0],
                "Percentage": (f"{round(dfn1['On Time'].tolist()[0] * 100, 2)}%"),
                "status": 'On Time',
            }

        return jsonify(success=True, data=stastic)
    else:
        return jsonify(success=False, errors=form.errors)


@app.route('/api/customer_payout', methods=['POST', 'GET'])
def customer_payout():
    Regressor_model = pickle.load(open(RF_Regressor_path, 'rb'))

    request_data = request.form
    account_id = request_data["account_id"]
    issue_month = request_data['issue_month']
    issue_day = request_data['issue_day']
    due_month = request_data['due_month']
    due_day = request_data['due_day']
    payment_terms = request_data['payment_terms']
    credit_history = request_data['credit_history']
    paid = request_data['paid']
    invoice_amount_main_currency = request_data['invoice_amount_main_currency']

    form = PayoutForm(request_data)
    if form.validate():
        X_COLUMS = ['Account Id', 'Issue Month', 'Issue Day', 'Due Month', 'Due Day', 'Payment Terms', 'Credit History',
                    'Paid', 'Invoice Amount Main Currency']
        Requests = np.array(
            [[account_id, issue_month, issue_day, due_month, due_day, payment_terms, credit_history, paid,
              invoice_amount_main_currency]])

        predicted = Regressor_model.predict(Requests)

        # Accuracy = r2_score(Requests, predicted)

        stastic = {
            "Days": predicted[0],
            # "Accuracy": Accuracy
        }

        return jsonify(success=True, data=stastic)
    else:
        return jsonify(success=False, errors=form.errors)


@app.route('/search', methods=['POST'])
def search():
    main_dictionary = pickle.load(open(main_dict_path, 'rb'))
    search_term = request.form.get('searchTerm', '').lower()
    search_results = []

    def recursive_search(d, parent_key=''):
        for key, value in d.items():
            if isinstance(value, dict):
                recursive_search(value, f'{parent_key} -> {key}' if parent_key else key)
            else:
                if search_term in str(key).lower() or search_term in str(value).lower():
                    full_key = f'{parent_key} -> {key}' if parent_key else key
                    search_results.append(f'{full_key}: {value}')

    recursive_search(main_dictionary)

    return jsonify({'results': search_results})


@app.route('/api/generateModel', methods=['POST'])
def json_to_model():
    try:
        method = 'pickle'

        if not os.path.exists(json_file_path):
            return jsonify ({"error": "JSON file not found"}), 404

        with open(json_file_path, 'r') as file:
            model_data = json.load(file)

        if method == 'pickle':
            with open(output_file_path, 'wb') as file:
                pickle.dump (model_data, file)
        elif method == 'joblib':
            joblib.dump(model_data, output_file_path)
        else:
            return jsonify({"error": "Invalid method. Choose 'pickle' or 'joblib'."}), 400

        return jsonify({"message": "Model saved successfully!", "path": output_file_path}), 200

    except Exception as e:
        return jsonify({"error": str (e)}), 500


if __name__ == '__main__':
    # app.run(debug=True)
    app.run(host='0.0.0.0')

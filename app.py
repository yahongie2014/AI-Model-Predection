import datetime
import os
import joblib
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn import preprocessing

from db import insert

app = Flask(__name__)

path = 'C://Users//ahmed.saeed//Desktop//FlaskApiRS//trainIncorta.pkl'
csv_path = 'C://Users//ahmed.saeed//Desktop//FlaskApiRS//RsIncorta.csv'
multi_csv_path = 'C:\\Users\\ahmed.saeed\\Desktop\\FlaskApiRS\\csv'


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


# Train Modle and Save In DB
@app.route('/save_db', methods=['GET'])
def dynamicDb():
    df_raw = []

    files = [os.path.join(multi_csv_path, f) for f in os.listdir(multi_csv_path) if
             os.path.isfile(os.path.join(multi_csv_path, f))]
    for x in files:
        df_raw = pd.read_csv(x, sep=',',
                             encoding='unicode_escape',
                             low_memory=False)
        df_raw.head(10)
        df_raw.columns = df_raw.columns.str.replace(' ', '_')
        df_raw.info()
        sum(df_raw.duplicated())
        df_raw.isnull().all()
        df_raw.isnull().sum()
        df1 = df_raw.dropna(subset=['Rs_Id',
                                    'Task_Type',
                                    'Brand',
                                    'Account',
                                    'unified_task_amount',
                                    'Unit',
                                    'Subject',
                                    'Tool',
                                    'Rs_Plan',
                                    'RS_Month',
                                    'RS_M_Day',
                                    'Hour',
                                    'Rs_WeekDay',
                                    'count_of_Resource',
                                    'Language_Pair'], how='any')
        df1
        label_encoder = preprocessing.LabelEncoder()
        df1["Brand"] = label_encoder.fit_transform(df1["Brand"])
        df1["Account"] = label_encoder.fit_transform(df1["Account"])
        df1["Unit"] = label_encoder.fit_transform(df1["Unit"])
        df1["Task_Type"] = label_encoder.fit_transform(df1["Task_Type"])
        df1["Subject"] = label_encoder.fit_transform(df1["Subject"])
        df1["Tool"] = label_encoder.fit_transform(df1["Tool"])
        df1["Rs_Plan"] = label_encoder.fit_transform(df1["Rs_Plan"])
        df1["Rs_WeekDay"] = label_encoder.fit_transform(df1["Rs_WeekDay"])
        df1["Language_Pair"] = label_encoder.fit_transform(df1["Language_Pair"])

        df1["Brand"] = pd.Categorical(df1["Brand"])
        df1["Account"] = pd.Categorical(df1["Account"])
        df1["Unit"] = pd.Categorical(df1["Unit"])
        df1["Task_Type"] = pd.Categorical(df1["Task_Type"])
        df1["Subject"] = pd.Categorical(df1["Subject"])
        df1["Tool"] = pd.Categorical(df1["Tool"])
        df1["Rs_Plan"] = pd.Categorical(df1["Rs_Plan"])
        df1["RS_Month"] = pd.Categorical(df1["RS_Month"])
        df1["RS_M_Day"] = pd.Categorical(df1["RS_M_Day"])
        df1["Hour"] = pd.Categorical(df1["Hour"])
        df1["Rs_WeekDay"] = pd.Categorical(df1["Rs_WeekDay"])
        df1["Language_Pair"] = pd.Categorical(df1["Language_Pair"])
        X = df1[
            ['Task_Type', 'Brand', 'unified_task_amount', 'DateStamp_base_2022', 'Duration', 'Unit', 'Subject', 'Tool',
             'Rs_Plan', 'RS_Month', 'RS_M_Day', 'Hour', 'Language_Pair', 'Rs_WeekDay']]
        y = np.array(df1["RS_Success"]).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
        rf_model = RandomForestRegressor(n_estimators=50, max_features="auto", random_state=44)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        meanAbsolute = mean_absolute_error(y_test, y_pred)
        meanSquared = mean_squared_error(y_test, y_pred)
        meanRoot = np.sqrt(mean_squared_error(y_test, y_pred))
        r2Score = r2_score(y_test, y_pred)
        sql = "INSERT INTO train_model(mean_absolute, mean_squared, root_mean ,filename,trained) VALUES (%s, %s,%s,%s,%s)"
        val = (meanAbsolute, meanSquared, meanRoot, x, 1)
        insert(sql, val)
    stastic = [
        {
            'Mean Absolute Error': meanAbsolute,
            'Mean Squared Error': meanSquared,
            'Root Mean Squared Error': meanRoot,
            'R2 Score': r2Score,
            'path': files
        }
    ]

    return jsonify(stastic)


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
        "Fail_Percentage": (f"{round(percentage[0][0] * 100, 2)}%"),
        "Success_Percentage": (f"{round(percentage[0][1] * 100, 2)}%"),
        "Prediction_Score": int(y_pred)
    }
    return jsonify(stastic)


if __name__ == '__main__':
    app.run(debug=True)

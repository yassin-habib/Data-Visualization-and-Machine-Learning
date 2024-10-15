from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train_model():
    dataset_file = request.files['file']
    dataset = pd.read_csv(dataset_file)
    dataset = dataset.drop("Time", axis=1)
    dataset = dataset.dropna()
    numerical_cols = dataset.drop(columns=['Class'])
    scaler = StandardScaler()
    dataset[numerical_cols.columns] = scaler.fit_transform(numerical_cols)
    joblib.dump(scaler,'scaler.pkl')
    X = dataset.drop(columns=['Class'])
    y = dataset['Class']
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    model_api = LogisticRegression()
    model_api.fit(X_train, y_train)
    y_pred = model_api.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    joblib.dump(model_api, 'model_weights_api.pkl')
    return jsonify({'precision': precision, 'recall': recall, 'f1_score': f1})


@app.route('/predict', methods=['POST'])
def make_predictions():
    new_data_file = request.files['file']
    new_data = pd.read_csv(new_data_file)
    new_data = new_data.drop(["Time", "Class"], axis=1)
    new_data = new_data.dropna()
    model = joblib.load('model_weights_api.pkl')
    scaler = joblib.load('scaler.pkl')
    predictions = predict_new_data(new_data, model, scaler)
    count_0 = np.count_nonzero(predictions == 0)
    count_1 = np.count_nonzero(predictions == 1)

    return jsonify({'predictions': predictions.tolist(), 'count_0': count_0,'count_1':count_1})


def predict_new_data(new_data, model, scaler):
    new_data_scaled = scaler.transform(new_data)
    predictions = model.predict(new_data_scaled)
    return predictions

@app.route('/')
def main_page():
    return render_template('temp.html')

if __name__ == '__main__':
    app.run(debug=True)
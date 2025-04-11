from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'mortality_model.pkl'

def generate_realistic_medical_data(n_samples=5000):
    """Genera datos médicos con relaciones realistas entre variables"""
    np.random.seed(42)
    
    # Edad: distribución normal con cola hacia mayores edades
    age = np.clip(np.random.normal(loc=50, scale=15, size=n_samples), 18, 100).astype(int)
    
    # Condiciones crónicas: distribución de Poisson (la mayoría tiene 0-2)
    chronic_conditions = np.minimum(np.random.poisson(1.2, size=n_samples), 5)
    
    # Presión arterial: correlacionada con edad y condiciones crónicas
    blood_pressure = np.clip(
        90 + age * 0.4 + chronic_conditions * 6 + np.random.normal(0, 8, n_samples),
        70, 190
    ).astype(int)
    
    # Oxígeno en sangre: inversamente correlacionado con condiciones crónicas
    oxygen_level = np.clip(
        98 - chronic_conditions * 3 - np.abs(np.random.normal(0, 3, n_samples)),
        75, 100
    ).astype(int)
    
    # Frecuencia cardíaca: correlacionada con presión arterial y edad
    heart_rate = np.clip(
        65 + (blood_pressure - 100) * 0.15 + (age - 50) * 0.1 + np.random.normal(0, 5, n_samples),
        45, 130
    ).astype(int)
    
    # Calcular severidad basada en múltiples factores
    severity = np.clip(
        1 + 
        (age > 65).astype(int) + 
        (blood_pressure > 140).astype(int) + 
        (oxygen_level < 90).astype(int) + 
        np.minimum(chronic_conditions, 3) +
        np.random.binomial(1, 0.2, size=n_samples),
        1, 5
    )
    
    # Calcular probabilidad de mortalidad de manera más realista
    mortality_prob = 1 / (1 + np.exp(
        -(-4.5 + 
         age * 0.03 + 
         (blood_pressure > 140) * 0.8 + 
         (oxygen_level < 90) * 1.2 + 
         chronic_conditions * 0.5)
    ))
    mortality = (np.random.rand(n_samples) < mortality_prob).astype(int)
    
    return pd.DataFrame({
        'age': age,
        'blood_pressure': blood_pressure,
        'heart_rate': heart_rate,
        'oxygen_level': oxygen_level,
        'chronic_conditions': chronic_conditions,
        'severity': severity,
        'mortality': mortality
    })

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Generar datos más realistas
    df = generate_realistic_medical_data(10000)
    
    # Dividir datos
    X = df.drop(['mortality', 'severity'], axis=1)
    y_mortality = df['mortality']
    
    # Entrenar modelo con mejores parámetros
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X, y_mortality)
    
    # Guardar modelo
    joblib.dump(model, MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validar datos de entrada
        if not all(key in data for key in ['age', 'blood_pressure', 'heart_rate', 'oxygen_level', 'chronic_conditions']):
            return jsonify({'status': 'error', 'message': 'Faltan parámetros requeridos'})
        
        # Preparar datos para predicción
        input_data = pd.DataFrame([{
            'age': max(18, min(100, int(data['age']))),
            'blood_pressure': max(70, min(190, int(data['blood_pressure']))),
            'heart_rate': max(40, min(130, int(data['heart_rate']))),
            'oxygen_level': max(70, min(100, int(data['oxygen_level']))),
            'chronic_conditions': max(0, min(5, int(data['chronic_conditions'])))
        }])
        
        # Hacer predicciones
        mortality_prob = model.predict_proba(input_data)[0][1]
        severity_pred = int(np.clip(
            1 + 
            (input_data['age'].values[0] > 65) + 
            (input_data['blood_pressure'].values[0] > 140) + 
            (input_data['oxygen_level'].values[0] < 90) + 
            min(input_data['chronic_conditions'].values[0], 3),
            1, 5
        ))
        
        # Ajustar probabilidad basada en severidad
        adjusted_prob = min(0.99, mortality_prob * (1 + severity_pred * 0.1))
        
        return jsonify({
            'mortality_probability': float(adjusted_prob),
            'severity_level': severity_pred,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

from datetime import datetime, timedelta
import json

# Simulación de base de datos para el histórico
HISTORY_DB = 'evaluations.json'

def load_evaluations():
    try:
        with open(HISTORY_DB, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_evaluation(evaluation):
    evaluations = load_evaluations()
    evaluations.append(evaluation)
    with open(HISTORY_DB, 'w') as f:
        json.dump(evaluations, f)

@app.route('/save_evaluation', methods=['POST'])
def save_eval():
    try:
        data = request.get_json()
        evaluation = {
            'id': f"eval-{datetime.now().timestamp()}",
            'timestamp': datetime.now().isoformat(),
            'patient_data': data['patient_data'],
            'results': data['results']
        }
        save_evaluation(evaluation)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/get_evaluations', methods=['GET'])
def get_evals():
    try:
        evaluations = load_evaluations()
        
        # Filtrar por fechas si se proporcionan
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        filtered = []
        for eval in evaluations:
            eval_date = datetime.fromisoformat(eval['timestamp'])
            
            if start_date:
                start = datetime.fromisoformat(start_date)
                if eval_date < start:
                    continue
            
            if end_date:
                end = datetime.fromisoformat(end_date)
                if eval_date > end:
                    continue
            
            filtered.append(eval)
        
        # Ordenar por fecha más reciente primero
        filtered.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'data': filtered,
            'count': len(filtered)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
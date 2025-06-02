import os
import json
import numpy as np
import pandas as pd
import joblib

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Count
from django.db.models.functions import TruncDate

from django.views.decorators.csrf import csrf_exempt
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import authentication_classes, permission_classes, api_view

from keras.models import load_model

from .models import Transaction, PredictionLog, CreditCardTransaction

# -------------------- ENVIRONMENT SETTINGS --------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# -------------------- MODEL & SCALER LOAD --------------------
credit_model = load_model(os.path.join(settings.BASE_DIR, 'detection/ml_models/credit_model.h5'))
credit_scaler = joblib.load(os.path.join(settings.BASE_DIR, 'detection/ml_models/credit_scaler.pkl'))

paysim_model = joblib.load(os.path.join(settings.BASE_DIR, 'detection/ml_models/final_Paysim_fraud_detection_xgb_model.pkl'))
paysim_scaler = joblib.load(os.path.join(settings.BASE_DIR, 'detection/ml_models/Paysim_scaler.pkl'))

online_model = joblib.load(os.path.join(settings.BASE_DIR, 'detection/ml_models/onlinefraud_detection_model.pkl'))
online_scaler = joblib.load(os.path.join(settings.BASE_DIR, 'detection/ml_models/onlinefraud_scaler.pkl'))

with open(os.path.join(settings.BASE_DIR, 'detection/ml_models/final_Paysim_best_threshold.txt'), 'r') as f:
    paysim_threshold = float(f.read())

_ = credit_model(np.zeros((1, 4)))  # Warm-up TensorFlow model

# -------------------- HOME VIEW --------------------
def home(request):
    return render(request, 'detection/home.html')

# -------------------- CREDIT CARD FRAUD --------------------
def creditcard_predict_view(request):
    prediction = None
    if request.method == 'POST':
        time_val = float(request.POST['time'])
        amount = float(request.POST['amount'])
        v10 = float(request.POST['v10'])
        v14 = float(request.POST['v14'])

        features = credit_scaler.transform([[time_val, amount, v10, v14]])
        result = credit_model.predict(features)[0][0]
        prediction = 'Fraud' if result > 0.5 else 'Legit'

        CreditCardTransaction.objects.create(
            time=time_val,
            amount=amount,
            v10=v10,
            v14=v14,
            prediction=prediction,
            timestamp=timezone.now()
        )

    return render(request, 'detection/form_credit.html', {'prediction': prediction})

# -------------------- Credit Dashboard --------------------
def dashboard_credit(request):
    transactions = CreditCardTransaction.objects.order_by('-timestamp')[:100]
    fraud_count = CreditCardTransaction.objects.filter(prediction='Fraud').count()
    legit_count = CreditCardTransaction.objects.filter(prediction='Legit').count()

    return render(request, 'detection/credit_dashboard.html', {
        'transactions': transactions,
        'fraud_count': fraud_count,
        'legit_count': legit_count,
    })

# -------------------- Creditcard Data JSON --------------------
def creditcard_data_json(request):
    transactions = CreditCardTransaction.objects.order_by('-timestamp')[:50]
    data = [
        {
            'time': tx.time,
            'amount': tx.amount,
            'prediction': tx.prediction
        }
        for tx in transactions
    ]
    return JsonResponse(data, safe=False)

# -------------------- PAYSIM FRAUD --------------------
def predict_paysim_fraud(request):
    if request.method == 'POST':
        try:
            amount = float(request.POST.get('amount'))
            oldbalanceOrg = float(request.POST.get('oldbalanceOrg'))
            newbalanceDest = float(request.POST.get('newbalanceDest'))
            oldbalanceDest = float(request.POST.get('oldbalanceDest'))
            txn_type = request.POST.get('type')

            if not txn_type:
                raise ValueError("Transaction type is missing.")

            type_features = ['PAYMENT', 'DEBIT', 'CASH_OUT', 'TRANSFER']
            type_encoded = {f'type_{t}': 1 if txn_type == t else 0 for t in type_features}

            input_data = {
                'amount': amount,
                'oldbalanceOrg': oldbalanceOrg,
                'newbalanceDest': newbalanceDest,
                'oldbalanceDest': oldbalanceDest,
                **type_encoded
            }

            input_df = pd.DataFrame([input_data])
            scaled_input = paysim_scaler.transform(input_df)

            probability = paysim_model.predict_proba(scaled_input)[0][1]
            prediction = 1 if probability > paysim_threshold else 0
            result = "Fraudulent" if prediction else "Not Fraudulent"

            Transaction.objects.create(
                timestamp=timezone.now(),
                amount=amount,
                tx_type=txn_type,
                prediction=prediction
            )

            return render(request, 'detection/paysim_result.html', {
                'prediction': result,
                'probability': f"{probability:.4f}",
                'model': 'PaySim_model'
            })

        except Exception as e:
            return render(request, 'detection/form_paysim.html', {'error': str(e)})

    return render(request, 'detection/form_paysim.html')

# -------------------- ONLINE PAYMENT FRAUD --------------------
@csrf_exempt
def predict_online_fraud(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            features = data.get('features')
            if features is None:
                raise ValueError("Missing 'features' in request body.")

            X = np.array(features).reshape(1, -1)
            X_scaled = online_scaler.transform(X)

            prediction = online_model.predict(X_scaled)
            predicted_label = int(prediction[0][0] > 0.5)

            PredictionLog.objects.create(
                model_name='Online_Model',
                prediction=predicted_label,
                input_data=json.dumps(features)
            )

            return JsonResponse({'prediction': predicted_label})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)

    return JsonResponse({'message': 'Only POST requests are allowed.'}, status=405)

def form_online(request):
    return render(request, 'detection/form_online.html')

# -------------------- DASHBOARD --------------------
def dashboard(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    label = request.GET.get('label')

    qs = Transaction.objects.all()
    if start_date:
        qs = qs.filter(timestamp__date__gte=start_date)
    if end_date:
        qs = qs.filter(timestamp__date__lte=end_date)
    if label in ('0', '1'):
        qs = qs.filter(prediction=int(label))

    chart_data_qs = (
        qs.filter(prediction=1)
        .annotate(date=TruncDate('timestamp'))
        .values('date')
        .annotate(count=Count('id'))
        .order_by('date')
    )

    chart_labels = [entry['date'].strftime('%Y-%m-%d') for entry in chart_data_qs]
    chart_data = [entry['count'] for entry in chart_data_qs]

    return render(request, 'detection/dashboard.html', {
        'predictions': qs.order_by('-timestamp')[:100],
        'chart_labels': chart_labels,
        'chart_data': chart_data
    })

# -------------------- REALTIME API --------------------
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def realtime_predict_api(request):
    try:
        data = request.data
        amount = float(data['amount'])
        oldbalanceOrg = float(data['oldbalanceOrg'])
        newbalanceDest = float(data['newbalanceDest'])
        oldbalanceDest = float(data['oldbalanceDest'])
        txn_type = data['type']

        type_features = ['PAYMENT', 'DEBIT', 'CASH_OUT', 'TRANSFER']
        type_encoded = {f'type_{t}': 1 if txn_type == t else 0 for t in type_features}

        input_data = {
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceDest': newbalanceDest,
            'oldbalanceDest': oldbalanceDest,
            **type_encoded
        }

        input_df = pd.DataFrame([input_data])
        scaled_input = paysim_scaler.transform(input_df)

        probability = paysim_model.predict_proba(scaled_input)[0][1]
        prediction = 1 if probability > paysim_threshold else 0

        Transaction.objects.create(
            timestamp=timezone.now(),
            amount=amount,
            tx_type=txn_type,
            prediction=prediction
        )

        return JsonResponse({
            'prediction': prediction,
            'probability': float(probability),
            'label': 'Fraudulent' if prediction else 'Not Fraudulent'
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

# -------------------- DASHBOARD API DATA --------------------
def dashboard_data(request):
    qs = (
        Transaction.objects.filter(prediction=1)
        .annotate(date=TruncDate('timestamp'))
        .values('date')
        .annotate(count=Count('id'))
        .order_by('date')
    )

    labels = [entry['date'].strftime('%Y-%m-%d') for entry in qs]
    counts = [entry['count'] for entry in qs]

    return JsonResponse({'labels': labels, 'counts': counts})

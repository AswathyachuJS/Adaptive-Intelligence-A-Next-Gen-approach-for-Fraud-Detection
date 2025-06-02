from django.urls import path
from . import views 

urlpatterns = [
    # Home Page
    path('', views.home, name='home'),

    # Prediction Endpoints (Forms & Logic)
    path('predict/credit/', views.creditcard_predict_view, name='predict_credit_card_fraud'),
    path('predict/paysim/', views.predict_paysim_fraud, name='predict_paysim_fraud'),
    path('predict/online/', views.predict_online_fraud, name='predict_online'),  # API-style POST

    # Online Fraud Manual Input Form
    path('form/online/', views.form_online, name='form_online'),

    # Dashboard View
    path('dashboard/', views.dashboard, name='dashboard'),
    
    path('dashboard-credit/json/', views.creditcard_data_json, name='dashboard_credit_json'),


    path('dashboard/credit/', views.dashboard_credit, name='dashboard_credit'),

    # Dashboard Chart Data API (for frontend JS/AJAX use)
    #path("api/dashboard-data/", views.dashboard_data, name="dashboard-data"),

    # Real-Time Prediction API (secured with Token Auth)
    path('realtime-predict/', views.realtime_predict_api, name='realtime_predict'),
    

]

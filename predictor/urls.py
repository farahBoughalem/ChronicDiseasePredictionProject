from django.urls import path
from .views import api_predict_diabetes_female, api_predict_diabetes_male, api_predict_hypertension, api_predict_cvd

urlpatterns = [
    path('api/diabetes/female/', api_predict_diabetes_female, name='api_predict_diabetes_female'),
    path('api/diabetes/male/', api_predict_diabetes_male, name='api_predict_diabetes_male'),
    path('api/hypertension/', api_predict_hypertension, name='api_predict_hypertension'),
    path('api/cvd/', api_predict_cvd, name='api_predict_cvd'),
]
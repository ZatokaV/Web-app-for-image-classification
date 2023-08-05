from django.apps import AppConfig


class PredictionAppConfig(AppConfig):
    """configuration class for the project, located in a submodule named apps.py"""
    default_auto_field: str = 'django.db.models.BigAutoField'
    name: str = 'prediction_app'


print(PredictionAppConfig.__doc__)

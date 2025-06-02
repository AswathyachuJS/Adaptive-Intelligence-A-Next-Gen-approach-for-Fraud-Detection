from django.db import models

class PredictionLog(models.Model):
    PREDICTION_CHOICES = [(0, 'Not Fraud'), (1, 'Fraud')]

    transaction_type = models.CharField(max_length=100)
    prediction = models.IntegerField(choices=PREDICTION_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.transaction_type} - {self.get_prediction_display()}"


class CreditCardTransaction(models.Model):
    time = models.FloatField()
    amount = models.FloatField()
    v10 = models.FloatField()
    v14 = models.FloatField()
    prediction = models.CharField(max_length=10)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.amount} - {self.prediction}"


class Transaction(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    amount = models.FloatField()
    tx_type = models.CharField(max_length=20)
    prediction = models.IntegerField(choices=[(0, 'Not Fraud'), (1, 'Fraud')])

    def __str__(self):
        return f"{self.tx_type} - {self.amount} - {'Fraud' if self.prediction else 'Not Fraud'}"

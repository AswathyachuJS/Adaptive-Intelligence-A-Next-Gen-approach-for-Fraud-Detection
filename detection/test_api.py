import requests

url = "http://127.0.0.1:8000/api/realtime-predict/"
headers = {
    "Authorization": "Token 9fc6d6539efb73fc4f689e718e689ca2c693df82",
    "Content-Type": "application/json"
}
data = {
    "amount": 1000,
    "oldbalanceOrg": 5000,
    "newbalanceDest": 3000,
    "oldbalanceDest": 1000,
    "type": "TRANSFER"
}

response = requests.post(url, headers=headers, json=data)
print("Status:", response.status_code)
print("Response:", response.json())

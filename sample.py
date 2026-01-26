import requests
import json

# API Configuration
URL = "http://127.0.0.1:8000/predict"

def test_35m_fraud_scenario():
    """
    Simulates a $35,000,000 transaction from a 'Clean' Identity.
    Expected Result: action='MANUAL_REVIEW_REQUIRED' or 'BLOCK_AND_CHALLENGE'
    """
    
    # Payload simulating a 'Clean' device but 'Extreme' amount
    payload = {
        "records": [
            {
                "TransactionAmt": 35000000.0,
                "TransactionDT": 86400,
                "DeviceInfo": "Windows",
                "id_31": "chrome 63.0",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "gmail.com",
                "id_30": "Windows 10",
                "DeviceType": "desktop"
            }
        ]
    }

    print(f"🚀 Sending Request: TransactionAmt = ${payload['records'][0]['TransactionAmt']:,}")
    
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
        result = response.json()

        print("\n--- API RESPONSE ---")
        print(json.dumps(result, indent=4))
        print("--------------------\n")

        # Verification Logic
        prob = result['probability']
        action = result['action']

        if action in ["BLOCK_AND_CHALLENGE", "MANUAL_REVIEW_REQUIRED"]:
            print(f"✅ SUCCESS: The system caught the outlier.")
            print(f"💡 Reason: Probability was {prob:.4f} and Action was {action}")
        else:
            print(f"❌ FAILURE: The system approved a $35M transaction.")

    except Exception as e:
        print(f"❌ Error connecting to API: {e}")

if __name__ == "__main__":
    test_35m_fraud_scenario()
"""
Week 7: API Testing Script

Tests the FastAPI churn prediction API endpoints.
"""

import requests

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint."""
    print("\n[Test 1] Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"  Status: {data['status']}")
            print(f"  Model Loaded: {data['model_loaded']}")
            print("  [OK]")
            return True
    except Exception as e:
        print(f"  [ERROR] {e}")
    return False


def test_single_prediction():
    """Test single prediction endpoint."""
    print("\n[Test 2] Single Prediction")
    
    customer = {
        "tenure": 1,
        "Contract": "Month-to-month",
        "MonthlyCharges": 99.0,
        "TotalCharges": 99.0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=customer)
        if response.status_code == 200:
            data = response.json()
            print(f"  Prediction: {'Churn' if data['churn_prediction']==1 else 'No Churn'}")
            print(f"  Probability: {data['churn_probability']:.2%}")
            print("  [OK]")
            return True
    except Exception as e:
        print(f"  [ERROR] {e}")
    return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Week 7: API Testing")
    print("=" * 60)
    print(f"API: {BASE_URL}")
    
    results = [
        ("Health", test_health()),
        ("Prediction", test_single_prediction())
    ]
    
    print("\n" + "=" * 60)
    passed = sum(1 for _, r in results if r)
    print(f"Results: {passed}/{len(results)} passed")


if __name__ == "__main__":
    main()

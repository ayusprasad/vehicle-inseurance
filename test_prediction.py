from src.components.vehicle_predictor import predict_insurance_response

# Test the prediction
result = predict_insurance_response(
    Gender="Male",
    Age=34,
    Driving_License=1,
    Region_Code=28.0,
    Previously_Insured=0,
    Vehicle_Age="< 1 Year",
    Vehicle_Damage="Yes",
    Annual_Premium=2630.0,
    Policy_Sales_Channel=26.0,
    Vintage=217
)

print("Prediction Result:")
print(f"Will buy insurance: {'YES' if result['prediction'] == 1 else 'NO'}")
print(f"Confidence: {result['probability']:.2%}")
# Mocking the gatekeeper with literal values from the prediction
result = {
    'confidence': 0.36,
    'displayConfidence': 36,
    'finalType': 'Karacadag',
    'success': True
}

result_conf = result.get('confidence', 1.0)
result_conf_pct = result.get('displayConfidence', 100)

print(f"result_conf: {result_conf}, type: {type(result_conf)}")
print(f"result_conf_pct: {result_conf_pct}, type: {type(result_conf_pct)}")
print(f"isinstance(result_conf, (int, float)): {isinstance(result_conf, (int, float))}")
print(f"result_conf < 0.75: {result_conf < 0.75}")

if (isinstance(result_conf, (int, float)) and result_conf < 0.75) or \
   (isinstance(result_conf_pct, (int, float)) and result_conf_pct < 75) or \
   (str(result.get('finalType', '')).lower() == 'non-rice'):
    print("GATEKEEPER FIRED: 422")
else:
    print("GATEKEEPER PASSED: 200")

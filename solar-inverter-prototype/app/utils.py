# app/utils.py
def missing_keys(required, provided):
    return [k for k in required if k not in provided]

def type_error_message(feature: str, expected: str):
    return f"{feature} must be {expected}"

def range_error_message(feature: str, lo, hi):
    return f"{feature} must be between {lo} and {hi}"
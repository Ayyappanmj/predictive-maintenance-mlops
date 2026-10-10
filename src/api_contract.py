"""Pydantic-free description of the API's input contract (shared by the API and promotion)."""

# API field name -> training column name
COLUMN_MAP = {
    "air_temperature_k": "Air temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]",
    "product_type": "Type",
}

EXAMPLE_READING = {
    "air_temperature_k": 300.1,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40.0,
    "tool_wear_min": 100,
    "product_type": "M",
}

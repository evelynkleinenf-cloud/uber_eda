from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "ncr_ride_bookings.csv"
MODELS_DIR = ROOT_DIR / "models"

DATE_CANDIDATES = [
    "Date",
    "pickup_datetime",
    "date",
    "datetime",
    "timestamp",
]

REGRESSION_TARGET_CANDIDATES = [
    "Booking Value",
    "fare_amount",
    "price",
    "fare",
    "amount",
]

CLASSIFICATION_TARGET_CANDIDATES = [
    "Booking Status",
    "trip_category",
    "ride_category",
    "target_class",
]

CLUSTER_FEATURE_CANDIDATES = [
    "Booking Value",
    "Ride Distance",
    "Avg VTAT",
    "Avg CTAT",
    "Driver Ratings",
    "Customer Rating",
    "hour",
    "month",
    "distance",
    "hour",
    "day_of_week",
    "month",
    "passenger_count",
    "fare_amount",
    "price",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]

MODEL_CATEGORICAL_FEATURES = [
    "Vehicle Type",
    "Payment Method",
    "Pickup Location",
    "Drop Location",
]

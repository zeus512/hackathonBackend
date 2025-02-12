from app import app  # Replace with your actual Flask file name
from flask_sse import sse
import redis

# Ensure Redis is running
r = redis.Redis(host="localhost", port=6379, db=0)

# Run inside Flask's app context
with app.app_context():
    sse.publish({"message": "Test"}, type="test_event")

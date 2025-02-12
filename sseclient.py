import requests
import json
import time

def test_sse_endpoint(audio_id, base_url="http://127.0.0.1:3000"):
    """Tests the SSE endpoint using requests."""

    sse_url = f"{base_url}/stream/chunk_update_{audio_id}"
    print(f"Connecting to SSE stream: {sse_url}")

    try:
        response = requests.get(sse_url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        print("SSE connection established.")

        for line in response.iter_lines():
            if line:  # Check for empty lines (keep-alive packets)
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]  # Remove "data: " prefix
                    print(f"Received event data: {data_str}")
                    try:
                        data = json.loads(data_str)
                        print(f"Event Data (JSON): {data}")
                    except json.JSONDecodeError:
                        print("Event data is not valid JSON")
                elif decoded_line.startswith(": ping"): # Handle ping messages
                    print("Received ping from server")
                else:
                    print(f"Received other message: {decoded_line}")


    except requests.exceptions.RequestException as e:
        print(f"Error connecting to SSE stream: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    audio_id_to_test = "main"  # Replace with your audio ID
    test_sse_endpoint(audio_id_to_test)
# # test_api.py

# import requests

# # Replace with the path to your image
# image_path = './img.jpg'

# # API endpoint
# url = 'http://localhost:5000/predict'

# # Open the image file in binary mode
# with open(image_path, 'rb') as img:
#     files = {'image': img}
#     response = requests.post(url, files=files)

# # Print the response from the server
# print(response.json())
# test_api.py

import requests

# Replace with the path to your image
image_path = './img2.jpg'  # Ensure this path is correct and the image exists

# API endpoint
url = 'http://localhost:5001/predict'

try:
    # Open the image file in binary mode
    with open(image_path, 'rb') as img:
        files = {'image': img}
        response = requests.post(url, files=files)

    # Print the status code
    print(f"Status Code: {response.status_code}")

    # Print the raw response text
    print(f"Response Text: {response.text}")

    # Try to parse JSON
    try:
        json_response = response.json()
        print("JSON Response:", json_response)
    except requests.exceptions.JSONDecodeError:
        print("Failed to decode JSON from response.")

except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

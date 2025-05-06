import requests

# Replace with the path to the image you want to test
image_path = 'path_to_your_image.jpg'  # Update this path to your image file

# URL of your FastAPI model deployed on Render
api_url = 'https://crop-disease-npbt.onrender.com/predict/'

# Open the image file in binary mode
with open(image_path, 'rb') as img_file:
    # Prepare the image file to be sent as part of the request
    files = {'file': ('image.jpg', img_file, 'image/jpeg')}
    
    # Send a POST request with the image file
    response = requests.post(api_url, files=files)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        result = response.json()  # Parse the response JSON
        print(f"Prediction result: {result['prediction']}")  # Print the predicted disease
    else:
        print(f"Failed to get prediction. Status code: {response.status_code}")

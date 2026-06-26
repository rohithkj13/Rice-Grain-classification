import requests

url = "http://localhost:5000/api/predict"
# Let's try sending a dummy image that looks like a stone or just whatever
# We can't easily grab the user's stone image, but we can make a dummy image
from PIL import Image
import io

img = Image.new('RGB', (200, 200), color = (100, 100, 100)) # Gray square
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

files = {'image': ('stone.jpg', img_bytes, 'image/jpeg')}
data = {'stable': 'true'}

try:
    response = requests.post(url, files=files, data=data)
    print("Status code:", response.status_code)
    print("Response JSON:", response.json())
except Exception as e:
    print("Error:", e)

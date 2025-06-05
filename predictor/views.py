from django.shortcuts import render

# Create your views here.
import base64
import io
import numpy as np
from PIL import Image
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import json

# Load model once
model = tf.keras.models.load_model('emotion_model.keras')
class_names = ['angry', 'happy', 'sad']

@csrf_exempt
def predict_emotion(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((48, 48))
        image_array = np.array(image).reshape(1, 48, 48, 1) / 255.0

        prediction = model.predict(image_array)
        emotion = class_names[np.argmax(prediction)]

        return JsonResponse({'emotion': emotion})
    else:
        return render(request, 'predictor/index.html')

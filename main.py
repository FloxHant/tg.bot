from keras.models  # TensorFlow is required for Keras to work
import load_model  # type: ignore
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import telebot # type: ignore
import os, random

def animal_classification(image_path):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model("keras_model.h5", compile=False)

    # Load the labels
    class_names = open("labels.txt", "r", encoding="utf-8").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open("depositphotos_13927400-stock-photo-chicken-isolated-on-white.jpg").convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name[2:-1]
animal = animal_classification("depositphotos_13927400-stock-photo-chicken-isolated-on-white.jpg")
print(animal)
bot = telebot.TeleBot('7783498381:AAE4y8ZEEmi4qIlvPKd_Pk9ob1Br9aeMxGI')

@bot.message_handler(commands=['animal'])
def send_meme(message):
    img_name =  random.choice(os.listdir('images'))
    with open(f'images/{img_name}', 'rb') as f:
        bot.send_photo(message.chat.id, f)


@bot.message_handler(commands=['start'])
def start_command(message):
    bot.send_message(message.chat.id, "Привет! Используй команду /ecology, чтобы получить интересный факт про экологию!")



bot.polling()

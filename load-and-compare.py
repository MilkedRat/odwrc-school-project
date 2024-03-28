import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import datetime
import os



# Get current timestamp to record when the prediction was made
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# User input for model and image file names
model_filename = input("Enter the filename of the trained model to load (e.g., 'my_model.h5'): ")
image_filename = input("Enter the filename of the image to classify: ")

# Load the model
model = load_model("models/" + model_filename)
print(f"Model {model_filename} loaded from disk.")

# Load and preprocess an image [Use the functions already defined in previous response]
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img) / 255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    
# Make predictions on a new image
def predict_image(model, img_array):
    predictions = model.predict(img_array)
    return np.argmax(predictions, axis=1)
    
    
# Define class names for CIFAR-10 [As per the previous response]
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Use the model to predict a new image
img_array = load_and_preprocess_image("traindata/" + image_filename)
predicted_class = predict_image(model, img_array)

# Print the result
print("Predicted class index:", predicted_class[0])
print("Predicted class name:", class_names[predicted_class[0]])

# Extract the base name of the model and image files (without the extension)
model_basename = os.path.splitext(model_filename)[0]
image_basename = os.path.splitext(os.path.basename(image_filename))[0]

# Create directories for logs if they don't exist
model_log_dir = os.path.join('logs', model_basename)
os.makedirs(model_log_dir, exist_ok=True)

# Path for the image-specific log within the model's log directory
image_log_file_path = os.path.join(model_log_dir, f"{image_basename}.txt")

# Log the prediction with a timestamp
with open(image_log_file_path, 'a') as file:
    file.write(f"{current_time}, Prediction: {class_names[predicted_class[0]]}\n")

# Inform the user that the prediction was logged
print(f"Prediction for {image_filename} has been logged in {image_log_file_path}.")
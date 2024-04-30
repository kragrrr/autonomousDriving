import os
import pandas as pd
import tensorflow as tf
import cv2

# Set the paths
data_folder = '/Users/krish/Development/SDC'
csv_file = '/Users/krish/Development/SDC/perf_Data/data.csv'

# Load the CSV file
data = pd.read_csv(csv_file)

# Load the model
model = tf.keras.models.load_model('models/model3.h5')

# Create an empty list to store the predicted angles
predicted_angles = []

def preProcess(img):
    img = img[60:135, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img

# Iterate through the image addresses
for image_address in data['Image_Address']:
    # Load the image
    image_path = os.path.join(data_folder, image_address)
    image = cv2.imread(image_path)
    image = preProcess(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])

    # Predict the angle using the loaded model
    predicted_angle = model.predict(image)[0][0]
    predicted_angles.append(predicted_angle)

# Add the predicted angles as the third column in the DataFrame
data['Predicted_Angle'] = predicted_angles

# Save the updated DataFrame to a new CSV file
output_csv = '/Users/krish/Development/SDC/perf_data/data_with_predicted_angles.csv'
data.to_csv(output_csv, index=False)

# Total Images Imported 4356
# Removed Images: 1730
# Remaining Images: 2626
# Training Samples: 2100
# Validation Samples: 526
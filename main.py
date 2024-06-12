import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
from skimage.metrics import peak_signal_noise_ratio as psnr

# Define a function to preprocess input data
def preprocess_input_image(image_path, target_size=(160, 240)):
    # Load and preprocess the image
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image) / 255.0  # Normalize the image
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

# Define a function to load the model
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Define a function to save the prediction
def save_prediction(image_array, save_path):
    image = array_to_img(image_array)
    save_img(save_path, image)

# Define the main function
def main():
    model_path = 'model.h5'  # Path to your saved model
    input_dir = './test/low/'
    output_dir = './test/predicted/'
    target_size = (160, 240)  # Adjust based on your model input size

    # Load the trained model
    model = load_trained_model(model_path)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each image in the input directory
    for image_name in os.listdir(input_dir):
        if image_name.endswith(('png', 'jpg', 'jpeg')):  # Add more extensions if needed
            input_image_path = os.path.join(input_dir, image_name)
            output_image_path = os.path.join(output_dir, image_name)
            
            # Preprocess the image
            input_image = preprocess_input_image(input_image_path, target_size)
            
            # Make prediction
            prediction = model.predict(input_image)
            
            # Post-process and save the prediction
            save_prediction(prediction[0], output_image_path)
            # Calculate PSNR
            original_image = img_to_array(load_img(input_image_path, target_size=target_size)) / 255.0
            predicted_image = prediction[0]
            psnr_value = psnr(original_image, predicted_image)
            print(f"PSNR for {image_name}: {psnr_value:.2f} dB")

    print(f"Predictions saved in {output_dir}")

if __name__ == "__main__":
    main()

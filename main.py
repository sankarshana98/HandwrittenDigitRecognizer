from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import os
import plotly
import plotly.graph_objs as go
import json
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam


UPLOAD_FOLDER = './static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model
model = load_model('mnist_newUp_model.h5', custom_objects={'Adam': Adam})

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Get the prediction details for the uploaded image
            prediction_details = predict(filepath)  
            
            feature_maps = get_feature_maps(filepath)
            # feature_map_images = save_feature_map_images(feature_maps, file.filename)
            feature_maps_paths = save_feature_map_images(feature_maps, file.filename)
            print("Saved feature maps to:", feature_maps_paths)



            return render_template('index.html', 
                                prediction_text='Predicted Number is: {}'.format(prediction_details['class_idx']),
                                confidence_text='Confidence: {:.2f}%'.format(prediction_details['confidence']*100),
                                file_name=file.filename,
                                feature_map_images=feature_maps_paths) # Pass filename for feedback

    return render_template('index.html', prediction_text='Upload an image of a handwritten digit.')


@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_value = request.form['feedback']
    image_file = request.form['image_file']  # Get the image filepath

    if feedback_value == "incorrect":
        correct_label = int(request.form['correct_label'])  

        # Load and preprocess the image
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file)
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img) / 255.0
        img_array = 1.0 - img_array

        # Fine-tune the model using the feedback
        fine_tune_model(img_array, correct_label, model)
    
    flash("Thanks for the feedback!")
    return redirect(url_for('upload_file'))

def predict(image_path):
    img = Image.open(image_path).convert('L')

    
    # Extract filename from image_path
    filename = os.path.basename(image_path)
    grayscale_path = os.path.join(app.config['UPLOAD_FOLDER'], 'grayscale_' + filename)

    
    # Save the resized grayscale image temporarily
    # grayscale_path = os.path.join('static', 'uploads', 'grayscale_' + filename)
    img.save(grayscale_path)

    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = 1.0 - img_array
    img_array = img_array.reshape(1, 28, 28, 1)
    
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    prediction_details = {
        'class_idx': class_idx,
        'confidence': predictions[0][class_idx],
        'predictions': predictions[0].tolist(),
        'graphJSON': create_plot(predictions[0]),
        'grayscale_path': grayscale_path  
    }
    return prediction_details

def create_plot(predictions):
    labels = list(range(10))
    data = [
        go.Bar(
            x=labels,
            y=predictions.tolist()
        )
    ]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

feature_map_model = Model(inputs=model.input, outputs=model.layers[0].output)
# Predict an image to get the feature maps
def get_feature_maps(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = 1.0 - img_array
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Get the feature maps
    feature_maps = feature_map_model.predict(img_array)
    return feature_maps[0]  # Return the first set (for the given image)

def fine_tune_model(image, correct_label, model):
    image = np.expand_dims(image, axis=0)
    label = to_categorical(correct_label, num_classes=10)
    label = np.expand_dims(label, axis=0)

    # Fine-tune the model using the feedback data
    model.train_on_batch(image, label)

    # Save the updated model
    save_model(model, 'mnist_newUp_model.h5')

def save_feature_map_images(feature_maps, base_filename):
    num_feature_maps = feature_maps.shape[-1]
    saved_paths = []

    for i in range(num_feature_maps):
        fm_image = feature_maps[:,:,i]
        
        # Normalize the feature map for better visualization
        fm_image = (fm_image - np.min(fm_image)) / (np.max(fm_image) - np.min(fm_image))
        
        base_filename_without_extension = os.path.splitext(base_filename)[0]
        filename = f"{base_filename_without_extension}_feature_map_{i}.png"
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        plt.imsave(path, fm_image, cmap='viridis')
        
        # Append just 'uploads/' + filename to saved_paths (relative to the static directory)
        saved_paths.append(os.path.join('uploads', filename))
    
    return saved_paths


if __name__ == '__main__':
    app.secret_key = 'super_secret_key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


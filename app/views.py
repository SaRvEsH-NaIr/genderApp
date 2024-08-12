from flask import render_template, request
import os 
import cv2
from app.face_recognition import faceRecognitionPipeline
import matplotlib.image as matimg

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')

def app():
    return render_template('app.html')

def genderApp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # Save our image in upload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) # Save image in upload folder
        # Get Predictions
        pred_image, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_Image.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}', pred_image)

        # Generate Report
        report = []
        for i, obj in  enumerate(predictions):
            gray_image = obj['roi'] # Gray Scale Image
            eigen_image = obj['eig_img'].reshape(100,100) # Eigen Image
            gender_name = obj['prediction_name'] # Name
            prob_score = round(obj['score']*100,2) # Probability

            # Save Gray Scale and eigen image
            gray_image_name = f'roi_{i}.jpg'
            eig_image_name = f'eigem_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}', gray_image, cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}', eigen_image, cmap='gray')

            # Save this report
            report.append([gray_image_name,eig_image_name,gender_name,prob_score])

        return render_template('gender.html', fileupload = True, report = report) # POST REQUEST

    return render_template('gender.html', fileupload = False) # GET REQUEST
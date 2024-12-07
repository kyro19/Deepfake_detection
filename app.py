import tensorflow as tf
import dlib
import cv2
import os
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Flask, render_template, request, redirect, send_file, url_for, Response
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import sqlite3

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def specificity_m(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def sensitivity_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (possible_positives + K.epsilon())
    return sensitivity

def mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

model_path2 = 'models/Xception.h5' 

custom_objects = {
    'f1_score': f1_score,
    'recall_m': recall_score,
    'precision_m': precision_score,
    'specificity_m': specificity_m,
    'sensitivity_m': sensitivity_m,
    'mae' : mae,
    'mse' : mse
}


model = load_model(model_path2, custom_objects=custom_objects)




@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/')
@app.route('/home')
def home():
	return render_template('home.html')

@app.route('/detection_results')
def detection_results():
	return render_template('detection_results.html')


@app.route('/upload')
def upload():
	return render_template('upload.html')

@app.route('/logon')
def logon():
	return render_template('signup.html')

@app.route('/login')
def login():
	return render_template('signin.html')


@app.route('/note')
def note():
	return render_template('note.html')

@app.route("/signup")
def signup():

    username = request.args.get('user','')
    name = request.args.get('name','')
    email = request.args.get('email','')
    number = request.args.get('mobile','')
    password = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
    con.commit()
    con.close()
    return render_template("signin.html")

@app.route("/signin")
def signin():

    mail1 = request.args.get('user','')
    password1 = request.args.get('password','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()

    if data == None:
        return render_template("signin.html")    

    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("index.html")

    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("index.html")
    else:
        return render_template("signup.html")



@app.route("/notebook")
def notebook():
    return render_template("notebook.html")




    
@app.route("/predict", methods=["GET", "POST"])
def predict_img():
    if request.method == "POST":
        if 'file' in request.files:
            f = request.files['file']
            filename = f.filename
            input_shape = (128, 128, 3)
            pr_data = []
            detector = dlib.get_frontal_face_detector()
            print("@@ Input posted = ", filename)
        
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)

            cap = cv2.VideoCapture(file_path)  
            frameRate = cap.get(5)

            predictions = []  

            while cap.isOpened():
                frameId = cap.get(1)
                ret, frame = cap.read()
                if ret != True:
                    break
                if frameId % ((int(frameRate) + 1) * 1) == 0:
                    face_rects, scores, idx = detector.run(frame, 0)
                    for i, d in enumerate(face_rects):
                        x1 = d.left()
                        y1 = d.top()
                        x2 = d.right()
                        y2 = d.bottom()
                        crop_img = frame[y1:y2, x1:x2]
                        data = img_to_array(cv2.resize(crop_img, (128, 128))).flatten() / 255.0
                        data = data.reshape(-1, 128, 128, 3)
                        prediction = model.predict(data)
                        prediction_class = np.argmax(prediction, axis=1)
                        predictions.append(prediction_class)
                        
            predictions = np.array(predictions)
            result = np.bincount(predictions.flatten()).argmax()

            if result == 0:
                ans = 'Fake'
            else:
                ans = 'Real'

              

            return render_template('display_image.html',  result=ans)

    return render_template('index.html')  

    




if __name__ == '__main__':
    app.run(debug=True)


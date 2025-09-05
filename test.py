import os
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
from datetime import datetime
  
sio=socketio.Server(always_connect = True)

app=Flask(__name__)
max_speed = 15


def  preprocessing(img):
    #will crop the image
    img=img[60:135,:,: ]
    img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img=cv2.GaussianBlur(img,(3,3),0)
    img=cv2.resize(img,(200,66))
    img=img/255

    return img
@sio.on('telemetry')
def telemetry(sid,data):
  
          #steering_angle = float(data["steering_angle"])
          #throttle = float(data["throttle"])
          speed = float(data["speed"])    
          image = Image.open(BytesIO(base64.b64decode(data["image"])))

          #timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
         # image_filename = os.path.join(r'path', timestamp)
          #image.save('{}.jpg'.format(image_filename))

          #try:
          image=np.asanyarray(image)
          image=preprocessing(image)
          image=np.array([image])
          predicted_values = model.predict(image)
          steering_angle = float(predicted_values)

            #global speed_limit
            #if speed > speed_limit:   
             #   speed_limit = min_speed
            #else:
             #   speed_limit = max_speed
          throttle = (1.0 - speed/max_speed)
   
          print('{} {} {}'.format(steering_angle,throttle,speed))
          sendcontrol(steering_angle,throttle)
          ''' except Exception as e:
               print (e)
     else:
        
        sio.emit('manual', data={}, skip_sid = True) '''


@sio.on('connect')
def connect(sid,environ):
    print("connect ", sid)
    sendcontrol(0,0) 


def sendcontrol(steering,throttle):
  sio.emit(
        "steer",
        data = {
            "steering_angle": steering.__str__(),
            "throttle": throttle.__str__()
        },
        skip_sid = True)

if __name__=='__main__':
    model=load_model('model.h5')
    app=socketio.Middleware(sio,app)
    eventlet.wsgi.server(eventlet.listen(('',4567)),app)
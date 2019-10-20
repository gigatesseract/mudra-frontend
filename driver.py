from flask import Flask, render_template, Response, jsonify
# from camera import VideoCamera
import cv2
import blink.detect_blinks as db
import gestureRecog.gestureCNN as gCNN
import serial
ser = serial.Serial()
print("connected to arduino uno... ")

from multiprocessing import Process, Value
app = Flask(__name__) 

map_index = {
  0:"Hello",
  1: "Right",
  2:"I love Hack-Inout",
  3: "Hello from Mudra",
  4:"Play",
  5:"Nothing"

}
x = input("Begin?")

state = ""

@app.route('/status', methods=['GET'])
def get_state():
  return jsonify({"state":state})



def infinite_loop():
  # arduino_tuple = (ser.read(1), ser.read(2), ser.read(3), ser.read(4))
  arduino_tuple = (25, 10, 392)
  global state
  # print(ser.read(1))
  cap = cv2.VideoCapture(0)
  ret = cap.set(3, 640)
  ret = cap.set(4, 480)
  flag = False
  framerate=0
  while(True):
    ret, frame = cap.read()
    flag = db.detect(flag, frame.copy())
    if(flag):
      framerate+=1
      print("flag on")
      preprocessed_image = gCNN.preprocess(frame.copy(), False, True)
      if(framerate%40==0):
       
        print(arduino_tuple)
        index = gCNN.predict(preprocessed_image.copy())
        state = map_index[index]
        print(state)
        
    else: 
      print("flag off")

    key = cv2.waitKey(5) & 0xff
    if key==ord('q'):
      break
  cap.release()
  cv2.destroyAllWindows()



if __name__ == "__main__":
  #  recording_on = Value('b', True)
   p = Process(target=infinite_loop)
   p.start()  
   app.run(host="0.0.0.0",debug=True, use_reloader=False)
   p.join()
# flag =False
# if db.detect(flag):


# app = Flask(__name__)


# video_stream = VideoCamera()

# @app.route('/')
# def index():
#     return render_template('index.html')

# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# @app.route('/video_feed')
# def video_feed():
#   return Response(gen(video_stream),mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=False,port="5000")
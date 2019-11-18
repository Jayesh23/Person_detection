from flask import Flask, render_template, Response
from yolo_bounding_box_image import get_frame
import yolo_bounding_box_image
import json, threading

t1 = threading.Thread(target=yolo_bounding_box_image.run_server)
t1.start()
t2 = threading.Thread(target=yolo_bounding_box_image.main_code)
t2.start()
print("reading....")

app = Flask(__name__)
	
@app.route('/')
def index():
	return render_template('index.html')
# y(1) & 0xFF == ord('q'):
	# 	break
def gen(capture):
	while True:
		# print("getting frame")
		frame = yolo_bounding_box_image.get_frame(capture)
		yield (b'--frame\r\n'
			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# def gen(camera):
# 	while True:
# 		print("getting frame")
# 		frame = camera.get_frame()
# 		yield (b'--frame\r\n'
# 			   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
	print(yolo_bounding_box_image.capture)
	return Response(gen(yolo_bounding_box_image.capture),
					mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
	app.run(use_reloader=False)
	app.run(host='127.0.0.1', port="5000", debug=True)

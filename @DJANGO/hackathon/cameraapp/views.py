from django.shortcuts import render

# Create your views here.
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import cv2
import main
import threading
from django.http import JsonResponse

# https://blog.miguelgrinberg.com/post/video-streaming-with-flask/page/8
cam = None


def home(request):
    context = {}
    return render(request, "cameraapp/camera-off.html", context)


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


def gen(camera):
    model = main.MyYolo.GET()
    while True:
        frame = model.process_cam(camera)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@gzip.gzip_page
def live(request):
    global cam
    try:
        cam = VideoCamera()
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print("에러입니다...")


def off(request):
    global cam
    if cam != None:
        cam.video.release()
        cv2.destroyAllWindows()
        cam = None
    return JsonResponse({'state': True})


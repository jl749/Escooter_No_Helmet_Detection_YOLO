from django.db import models

# Create your models here.


class Video(models.Model):
    title = models.CharField(max_length=200)
    video_file = models.FileField(upload_to='videos/', null=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    yolo_file = models.FileField(upload_to='yolovideos/',null=True)







from django import forms
from django.forms import ModelForm

from videoapp.models import Video


class VideoCreationForm(ModelForm):

    class Meta:
        model = Video
        fields = ['title', 'video_file']
from django.urls import path

from cameraapp.views import live, home, off

app_name = 'cameraapp'

urlpatterns = [
    path('', home, name='home'),
    path('live/', live, name='camera'),
    # path('on/', on, name='on'),
    path('off/', off, name='off')
]
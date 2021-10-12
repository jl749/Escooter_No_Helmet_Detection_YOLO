from django.urls import path

from videoapp.views import VideoListView, VideoCreateView, VideoDetailView, hello_helmet

app_name = 'videoapp'

urlpatterns = [
    path('', hello_helmet, name='home'),
    path('list/', VideoListView.as_view(), name='list'),
    path('create/', VideoCreateView.as_view(), name='create'),
    path('detail/<int:pk>', VideoDetailView.as_view(), name='detail'),
]
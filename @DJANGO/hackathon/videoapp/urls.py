from django.urls import path

from videoapp.views import VideoListView, VideoCreateView, VideoDetailView

app_name = 'videoapp'

urlpatterns = [
    path('list/', VideoListView.as_view(), name='list'),
    path('create/', VideoCreateView.as_view(), name='create'),
    path('detail/<int:pk>', VideoDetailView.as_view(), name='detail'),
]
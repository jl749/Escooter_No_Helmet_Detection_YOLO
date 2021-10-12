from django.shortcuts import render

# Create your views here.
from django.urls import reverse_lazy, reverse
from django.views.generic import ListView, CreateView, DetailView
from django.views.generic.edit import FormMixin

from videoapp.forms import VideoCreationForm
from videoapp.models import Video
import main


class VideoCreateView(CreateView):
    model = Video
    form_class = VideoCreationForm
    #success_url = reverse_lazy('videoapp:list')  # create 가 성공적으로 수행이 되면 yolo 에 detail view 로 연결
    template_name = 'videoapp/create.html'

    def get_success_url(self):
        v = self.model.objects.get(pk=self.object.pk)
        videofile = v.video_file
        yolo = main.MyYolo.GET()
        yolo.processVideo('media/'+str(videofile),'media/yolovideos'+str(videofile)[6:])
        #main.MyYolo().processVideo(str(videofile),'media/yolovideos'+str(videofile)[6:])
        v.yolo_file = 'media/yolovideos'+str(videofile)[6:]
        v.save()
        return reverse('videoapp:detail', kwargs={'pk': self.object.pk})

class VideoDetailView(DetailView):
    model = Video
    context_object_name = 'target_video'
    template_name = 'videoapp/detail.html'


    # button 을 누르면 yolomodel 과정 통과







class VideoListView(ListView):
    model = Video
    context_object_name = 'video_list'
    template_name = 'videoapp/list.html'
    paginate_by = 10

from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('techniques/', views.list_techniques, name='list_techniques'),
    path('techniques/<str:technique_id>/', views.get_technique, name='get_technique'),
    path('videos/upload/', views.upload_video, name='upload_video'),
    path('videos/<str:file_id>/summarize/', views.summarize_video, name='summarize_video'),
    path('videos/<str:file_id>/analytics/', views.get_analytics, name='get_analytics'),
    path('videos/<str:file_id>/compare/', views.compare_techniques, name='compare_techniques'),
    path('videos/<str:file_id>/insights/', views.ai_insights, name='ai_insights'),
    path('videos/<str:file_id>/', views.delete_video, name='delete_video'),
]

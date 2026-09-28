from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve
from django.views.generic import TemplateView

urlpatterns = [
    path('', TemplateView.as_view(template_name='index.html'), name='home'),
    path('api/', include('api.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])

# Uploads and summaries are written at runtime, so WhiteNoise (which indexes
# STATIC_ROOT at startup) can't serve them. django.conf.urls.static.static() is a
# no-op when DEBUG=False, so wire the serve view directly. It uses safe_join, so
# path traversal is still blocked. Fine for a single-instance deployment; put
# nginx or object storage in front of these if this ever scales out.
urlpatterns += [
    re_path(r'^uploads/(?P<path>.*)$', serve, {'document_root': settings.MEDIA_ROOT}),
    re_path(r'^outputs/(?P<path>.*)$', serve, {'document_root': settings.OUTPUT_ROOT}),
]

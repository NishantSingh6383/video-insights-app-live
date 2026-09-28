import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'dev-secret-key-change-in-production')

DEBUG = os.environ.get('DJANGO_DEBUG', 'True').lower() == 'true'

# In production, set DJANGO_ALLOWED_HOSTS to a comma-separated list of hostnames.
ALLOWED_HOSTS = [
    h.strip()
    for h in os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
    if h.strip()
]

# Render injects the service's public hostname at runtime; it isn't known at deploy time.
RENDER_EXTERNAL_HOSTNAME = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
if RENDER_EXTERNAL_HOSTNAME and RENDER_EXTERNAL_HOSTNAME not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

CSRF_TRUSTED_ORIGINS = [
    f'https://{h}' for h in ALLOWED_HOSTS
    if h not in ('localhost', '127.0.0.1', '*') and not h.startswith('.')
]

INSTALLED_APPS = [
    'django.contrib.contenttypes',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'

DATABASES = {}

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

# WhiteNoise serves collected static files straight from gunicorn, so the free
# tier doesn't need a separate nginx/CDN in front of the app.
STORAGES = {
    'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'},
    'staticfiles': {'BACKEND': 'whitenoise.storage.CompressedManifestStaticFilesStorage'},
}

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'http://127.0.0.1:3000',
    'http://localhost:5173',
    'http://127.0.0.1:5173',
]

REST_FRAMEWORK = {
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'UNAUTHENTICATED_USER': None,
}

MEDIA_ROOT = BASE_DIR.parent / 'uploads'
OUTPUT_ROOT = BASE_DIR.parent / 'outputs'

MEDIA_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_SIZE = int(os.environ.get('MAX_UPLOAD_MB', '500')) * 1024 * 1024

# Summarization loads every sampled frame into memory at once. On a small host
# (e.g. Render's 512MB free tier) set SUMMARY_MAX_FRAME_WIDTH=640 and
# SUMMARY_MAX_EXTRACT_FRAMES=200 to keep peak memory well under the limit.
# 0 / unset preserves the original full-resolution behaviour for local runs.
SUMMARY_MAX_FRAME_WIDTH = int(os.environ.get('SUMMARY_MAX_FRAME_WIDTH', '0'))
SUMMARY_MAX_EXTRACT_FRAMES = int(os.environ.get('SUMMARY_MAX_EXTRACT_FRAMES', '500'))

# The analytics endpoints hold their sampled frames in one array. Peak memory is
# roughly width * (width*9/16) * 3 * frames; at 480px and 300 frames that is
# ~117MB per request, and the dashboard calls analytics and compare back to back.
# Lower these on a small instance so the two together cannot exhaust it.
ANALYTICS_FRAME_WIDTH = int(os.environ.get('ANALYTICS_FRAME_WIDTH', '480'))
ANALYTICS_MAX_FRAMES = int(os.environ.get('ANALYTICS_MAX_FRAMES', '1000'))

# Analytics and comparison results are pure functions of an immutable upload
# (file_id is a UUID and the file never changes), so they cache safely. Without
# this, every visit to the analytics tab re-decoded the video and re-ran optical
# flow. LocMem keeps it dependency-free; entries are small JSON blobs.
ANALYTICS_CACHE_TTL = int(os.environ.get('ANALYTICS_CACHE_TTL', '3600'))

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'video-insights',
        'TIMEOUT': ANALYTICS_CACHE_TTL,
        'OPTIONS': {'MAX_ENTRIES': 64, 'CULL_FREQUENCY': 4},
    }
}

CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Security hardening (active when DEBUG=False behind HTTPS)
if not DEBUG:
    # TLS terminates at the platform proxy (Render/Heroku/nginx), which forwards
    # the original scheme. Without this Django treats every request as plain HTTP
    # and SECURE_SSL_REDIRECT would loop forever.
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
    SECURE_SSL_REDIRECT = os.environ.get('DJANGO_SECURE_SSL_REDIRECT', 'True').lower() == 'true'
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 31536000
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True

SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {'format': '[{levelname}] {asctime} {name}: {message}', 'style': '{'},
    },
    'handlers': {
        'console': {'class': 'logging.StreamHandler', 'formatter': 'simple'},
    },
    'root': {'handlers': ['console'], 'level': 'INFO'},
}

from django.urls import re_path
from .consumers import MyConsumer
from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter

websocket_urlpatterns = [
       re_path(r'ws/socket-server/',MyConsumer.as_asgi()),
]

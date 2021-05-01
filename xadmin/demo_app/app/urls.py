# -*- coding: utf-8 -*-
# from django.conf.urls import include, url
from django.urls import include, path
from demo_app.app import rest
# Uncomment the next two lines to enable the admin:
import xadmin

xadmin.autodiscover()

# version模块自动注册需要版本控制的 Model
from xadmin.plugins import xversion

xversion.register_models()

from django.contrib import admin

urlpatterns = [
    path('xadmin/', xadmin.site.urls),
    path('login/', rest.Login, name='login'),
    path('register/', rest.Register, name='register'),
    path('checkImage/', rest.CheckImage, name='checkImage'),
    path('checkVideo/', rest.CheckVideo, name='checkVideo'),
    path('collect/', rest.CollectData, name='collect'),
    path('getMyCollect/', rest.GetMyCollect, name='getMyCollect'),
    path('getMyHistory/', rest.GetMyHistory, name='getMyHistory'),
    path('getVerifyCode/', rest.GetVerifyCode, name='getVerifyCode'),
    path('changePwd/', rest.ChangePwd, name='changePwd'),
    path('deleteCollection/', rest.deleteCollection, name='deleteCollection'),
    path('deleteHistory/', rest.deleteHistory, name='deleteHistory'),
    path('deleteAllCollection/', rest.deleteAllCollection, name='deleteAllCollection'),
    path('deleteAllHistory/', rest.deleteAllHistory, name='deleteAllHistory'),
]

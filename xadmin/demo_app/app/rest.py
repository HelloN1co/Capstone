import json
from django.http import HttpResponse, JsonResponse
import hashlib
from .models import IDC, Host, MaintainLog, HostGroup, AccessRecord, Choice, Poll, Member, History, Collect
import json
import os
import sys
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import random
from demo_app.app.yolo_video import check_image
from demo_app.app.yolo import YOLO, check_video
import time
import datetime
from demo_app.app.obj import His, Col
import cv2
import _thread
from django.core import serializers
from django.db import connection
from queue import Queue

temp = os.getcwd() + "/app/"
# 先进先出队列
q = Queue(maxsize=5000)



@csrf_exempt
def Login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        print(username + ":" + password)
        md5 = hashlib.md5()
        md5.update(password.encode("utf-8"))
        pwd = md5.hexdigest()
        list = Member.objects.raw("SELECT * from member where email='" + username + "' and password='" + pwd + "'")
        if list != None and len(list) > 0:
            m = list[0]
            return JsonResponse({"code": 200, "data": m.id, "email": m.email}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 200}, safe=False)


@csrf_exempt
def Register(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        print(username + ":" + password)
        md5 = hashlib.md5()
        md5.update(password.encode("utf-8"))
        pwd = md5.hexdigest()
        list = Member.objects.raw("SELECT * from member where email='" + username + "'")
        if list != None and len(list) > 0:
            return JsonResponse({"code": 300}, safe=False)
        else:
            member = Member()
            member.password = pwd
            member.email = username
            member.save()
            return JsonResponse({"code": 200}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def GetVerifyCode(request):
    if request.method == "POST":
        email = request.POST.get("username")
        code = sendCodeToEmail(email)
        print('code=' + code)
        if len(code) == 0:
            return JsonResponse({"code": 400}, safe=False)
        else:
            return JsonResponse({"code": 200, "data": code}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)

@csrf_exempt
def ChangePwd(request):
    if request.method == "POST":
        mid = request.POST.get("mid")
        password = request.POST.get("password")
        md5 = hashlib.md5()
        md5.update(password.encode("utf-8"))
        pwd = md5.hexdigest()
        Member.objects.filter(id=mid).update(password=pwd)
        return JsonResponse({"code": 200}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)

def sendCodeToEmail(target):
    my_sender = '825849120@qq.com'
    my_pass = 'uehnckkqidhvbeib'
    ret = True
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    c = random.randint(0, 9)
    d = random.randint(0, 9)
    code = str(a) + str(b) + str(c) + str(d)
    try:
        msg = MIMEText('Welcome to use our system. Your verification code is: ' + code, 'plain', 'utf-8')
        msg['From'] = formataddr(["From detection system", my_sender])
        msg['To'] = formataddr(["FK", target])
        msg['Subject'] = "Verification Code"
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # sender's smtp server
        server.login(my_sender, my_pass)  # sender login
        server.sendmail(my_sender, [target, ], msg.as_string())  # send email to target
        server.quit()  # close
    except Exception as e:
        ret = False
    if ret == True:
        return code
    else:
        return ""


def doCheckImage(history):
    imgFile = history.srcFilePath
    print("check image^^^^^^^^^^^^^^^^^^^^^^^^^^^^^" + imgFile)
    t = time.time()
    yolo = YOLO()
    destFile = os.getcwd() + "/app/static/dest/" + str(int(round(t * 1000))) + ".jpg"
    check_image(yolo, imgFile, destFile)
    history.destFilePath = destFile.replace(temp, '')
    history.result = "complete"
    history.thumnail = destFile.replace(temp, '')
    history.save()
    print("finish")


def doCheckVideo(history):
    print("check video^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    videoFile = history.srcFilePath
    t = time.time()
    yolo = YOLO()
    videoDestFile = os.getcwd() + "/app/static/videos/" + str(int(round(t * 1000))) + ".MP4"
    check_video(yolo, videoFile, videoDestFile)
    vidcap = cv2.VideoCapture(videoDestFile)
    success, image = vidcap.read()
    n = 1
    while n < 2:
        success, image = vidcap.read()
        n += 1
    thumnail = os.getcwd() + "/app/static/dest/" + str(int(round(t * 1000))) + ".jpg"
    imag = cv2.imwrite(thumnail, image)
    if imag == True:
        print('finish')
        history.result = "complete"
        history.thumnail = "/" + thumnail.replace(temp, '')
        history.destFilePath = "/" + videoDestFile.replace(temp, '')
        history.srcFilePath = "/" + videoFile.replace(temp, '')
    history.save()


@csrf_exempt
def CheckImage(request):
    if request.method == "POST":
        mid = request.POST.get("mid")
        files = request.FILES.getlist('files')
        cur = os.getcwd()
        imgDir = os.getcwd() + "/app/static/images/"
        create_dir_not_exist(imgDir)
        for f in files:
            imgFile = imgDir + f.name
            destination = open(imgFile, 'wb+')
            print(imgFile + "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            for chunk in f.chunks():
                destination.write(chunk)
            destination.close()
            history = History()
            history.mid = mid
            history.category = 'Image'
            history.name = f.name
            history.srcFilePath = imgFile
            history.destFilePath = ''
            history.thumnail = ''
            history.result = ''
            q.put(history)
            # doCheckImage(history)

        return JsonResponse({"code": 200}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


def create_dir_not_exist(path):
    if os.path.exists(path) != True:
        os.makedirs(path)


@csrf_exempt
def CheckVideo(request):
    if request.method == "POST":
        file_obj = request.FILES.get('file')
        cur = os.getcwd() + "/app/static/videos/"
        video = cur + file_obj.name
        f = open(video, 'wb')
        print(file_obj, type(file_obj))
        for chunk in file_obj.chunks():
            f.write(chunk)
        f.close()
        mid = request.POST.get("mid")
        history = History()
        history.mid = mid
        history.category = 'Video'
        history.name = file_obj.name
        history.srcFilePath = video
        history.destFilePath = ''
        history.thumnail = ''
        history.result = ''
        # doCheckVideo(history)
        q.put(history)
        return JsonResponse({"code": 200}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def CollectData(request):
    if request.method == "POST":
        mid = request.POST.get("mid")
        hid = request.POST.get("hid")
        id = int(hid)
        ls = History.objects.raw("SELECT * from history where id=" + str(id) + "")
        if ls != None or len(ls) > 0:
            his = ls[0]
            collect = Collect()
            collect.hid = hid;
            collect.mid = mid
            collect.category = his.category
            collect.name = his.name
            collect.srcFilePath = his.srcFilePath
            collect.destFilePath = his.destFilePath
            collect.thumnail = his.thumnail
            collect.result = his.result
            collect.save()
            return JsonResponse({"code": 200}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def GetMyCollect(request):
    if request.method == "POST":
        id = request.POST.get("mid")
        page = request.POST.get("page")
        size = request.POST.get("pageSize")
        dt = request.POST.get("time")
        offset = (int(page) - 1) * int(size)
        sql = "SELECT DATE_FORMAT(create_time,'%m-%d-%Y') time from collect where mid=" + str(id) + " group by DATE_FORMAT(create_time,'%m-%d-%Y')"
        cursor = connection.cursor()
        cursor.execute(sql)
        row = cursor.fetchall()
        times = []
        print(row)
        if row != None or len(row) > 0:
            for item in row:
                times.append(item[0])
        tts = json.dumps(times)
        ft = '%%m-%%d-%%Y'
        if dt != None and len(dt) > 0:
            sql = "SELECT * from collect where mid=%d and DATE_FORMAT(create_time,'%s')='%s' order by create_time desc limit %d,%d" % (
                int(id), ft, dt, offset, int(size))
        else:
            sql = "SELECT * from collect where mid={0} order by create_time desc limit {1},{2}".format(id, offset,
                                                                                                       int(size))

        data = Collect.objects.raw(sql)
        if data != None or len(data) > 0:
            ls = []
            if dt != None and len(dt) > 0:
                sql = "SELECT * from collect where mid=%d and DATE_FORMAT(create_time,'%s')='%s' order by create_time desc " % (
                    int(id), ft, dt)
            else:
                sql = "SELECT * from collect where mid={0} order by create_time desc ".format(id)
            print(sql)
            dd = Collect.objects.raw(sql)
            count = len(dd)
            for item in data:
                item.create_time = str(item.create_time)
                collect = Col()
                collect.id = item.id
                collect.mid = item.mid
                collect.category = item.category
                collect.name = item.name
                collect.srcFilePath = item.srcFilePath
                collect.destFilePath = item.destFilePath
                collect.thumnail = item.thumnail
                collect.result = item.result
                collect.create_time = item.create_time
                ls.append(collect.__dict__)
            dt = json.dumps(ls)
            return JsonResponse({"code": 200, "count": count, "data": dt, "date": tts}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def GetMyHistory(request):
    if request.method == "POST":
        id = request.POST.get("mid")
        page = request.POST.get("page")
        size = request.POST.get("pageSize")
        dt = request.POST.get("time")
        offset = (int(page) - 1) * int(size)
        sql = "SELECT DATE_FORMAT(create_time,'%m-%d-%Y') time from history where mid=" + str(id) + " group by DATE_FORMAT(create_time,'%m-%d-%Y')"
        cursor = connection.cursor()
        cursor.execute(sql)
        row = cursor.fetchall()
        times = []
        print(row)
        if row != None or len(row) > 0:
            for item in row:
                times.append(item[0])
        tts = json.dumps(times)
        ft = '%%m-%%d-%%Y'
        if dt != None and len(dt) > 0:
            sql = "SELECT * from history where mid=%d and DATE_FORMAT(create_time,'%s')='%s' order by create_time desc limit %d,%d" % (
                int(id), ft, dt, offset, int(size))
        else:
            sql = "SELECT * from history where mid={0} order by create_time desc limit {1},{2}".format(id, offset,
                                                                                                       int(size))

        data = History.objects.raw(sql)
        if data != None or len(data) > 0:
            ls = []
            if dt != None and len(dt) > 0:
                sql = "SELECT * from history where mid=%d and DATE_FORMAT(create_time,'%s')='%s' order by create_time desc " % (
                    int(id), ft, dt)
            else:
                sql = "SELECT * from history where mid={0} order by create_time desc ".format(id)
            print(sql)
            dd = History.objects.raw(sql)
            count = len(dd)
            for item in data:
                item.create_time = str(item.create_time)
                history = His()
                history.id = item.id
                history.mid = item.mid
                history.category = item.category
                history.name = item.name
                history.srcFilePath = item.srcFilePath
                history.destFilePath = item.destFilePath
                history.thumnail = item.thumnail
                history.result = item.result
                history.create_time = item.create_time
                ls.append(history.__dict__)
            dt = json.dumps(ls)
            return JsonResponse({"code": 200, "count": count, "data": dt, "date": tts}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def deleteCollection(request):
    if request.method == "POST":
        id = request.POST.get("id")
        sql = "DELETE from collect where id=" + str(id) + ""
        cursor = connection.cursor()
        cursor.execute(sql)

        ls = Collect.objects.raw(sql)
        if ls != None or len(ls) > 0:
            return JsonResponse({"code": 200}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def deleteHistory(request):
    if request.method == "POST":
        id = request.POST.get("id")
        sql = "DELETE from history where id=" + str(id) + ""
        cursor = connection.cursor()
        cursor.execute(sql)

        ls = History.objects.raw(sql)
        if ls != None or len(ls) > 0:
            return JsonResponse({"code": 200}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def deleteAllCollection(request):
    if request.method == "POST":
        mid = request.POST.get("mid")
        sql = "DELETE from collect where mid=" + str(mid) + ""
        cursor = connection.cursor()
        cursor.execute(sql)

        ls = Collect.objects.raw(sql)
        if ls != None or len(ls) > 0:
            return JsonResponse({"code": 200}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


@csrf_exempt
def deleteAllHistory(request):
    if request.method == "POST":
        mid = request.POST.get("mid")
        sql = "DELETE from history where mid=" + str(mid) + ""
        cursor = connection.cursor()
        cursor.execute(sql)

        ls = History.objects.raw(sql)
        if ls != None or len(ls) > 0:
            return JsonResponse({"code": 200}, safe=False)
        else:
            return JsonResponse({"code": 300}, safe=False)
    else:
        return JsonResponse({"code": 400}, safe=False)


def monitorFunc(num):
    while True:
        if q.empty() == True:
            time.sleep(10)
        else:
            history = q.get()
            if history.category == 'Image':
                print("check image:"+history.srcFilePath)
                doCheckImage(history)
            else:
                print("check video:" + history.srcFilePath)
                doCheckVideo(history)

try:
    _thread.start_new_thread(monitorFunc, (1,))
    print("Start monitor thread----------------------->")
except:
    print("Error: cannot start")

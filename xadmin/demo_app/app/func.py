
import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr
import random
import os
from demo_app.app.yolo_video import check_image
from demo_app.app.yolo import YOLO, check_video


def sendCodeToEmail(target):
    my_sender = '825849120@qq.com'
    my_pass = 'uehnckkqidhvbeib'
    ret = True
    a = random.randint(0, 9)
    b = random.randint(0, 9)
    c = random.randint(0, 9)
    d = random.randint(0, 9)
    code = str(a)+str(b)+str(c)+str(d)
    try:
        msg = MIMEText('Verification Code is:'+code, 'plain', 'utf-8')
        msg['From'] = formataddr(["From nicead.top", my_sender])  # 括号里的对应发件人邮箱昵称、发件人邮箱账号
        msg['To'] = formataddr(["FK", target])
        msg['Subject'] = "Verification Code"
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 发件人邮箱中的SMTP服务器，端口是25
        server.login(my_sender, my_pass)  # 括号中对应的是发件人邮箱账号、邮箱密码
        server.sendmail(my_sender, [target, ], msg.as_string())  # 括号中对应的是发件人邮箱账号、收件人邮箱账号、发送邮件
        server.quit()  # 关闭连接
    except Exception as e:  # 如果 try 中的语句没有执行，则会执行下面的ret=False
        ret = False
    if ret == True:
        return code
    else:
        return ""


def main():
    print('this message is from main function')
    # sendCodeToEmail('hslb1985@163.com')
    # imgFile = os.getcwd()+"/1.jpg"
    # destFile = os.getcwd()+"/2.jpg"
    # check_image(imgFile,destFile)

    videoFile = os.getcwd() + "/haha.mp4"
    videoDestFile = os.getcwd() + "/dest.mp4"
    check_video(YOLO(),videoFile,videoDestFile)

if __name__ == '__main__':
    main()
    # print(__name__)
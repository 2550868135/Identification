import requests
import random
from random import choice
import time

from lxml import etree

GLOBAL_NAME = set()
FIRST_NAME = ["赵","钱","孙","李","周","吴","郑","张","王","朱","蔡","杨","丁","马"]
LAST_NAME = ["宇","帅","劲","翔","伟","志","佳","俊","新","纪","凯","昊","天","行","一","昆"]
# 字体选择列表
FONT_INFO_ID = [567, 532, 219, 270]
# 其他固定参数
FONT_SIZE = 52
FONT_COLOR = "#000000"
IMAGE_WIDTH = 570
IMAGE_HEIGHT = 120
IMAGE_BG_COLOR = "#FFFFFF"
ACTION_CATEGORY = 2

HEADERS = {"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36",
           "Referer": "http://www.diyiziti.com/qianming",
           "Host": "www.diyiziti.com",
           "Cookie": "Hm_lvt_1a6016c8e736ecef523fbe2539419b5a=1622381109,1622686077; Hm_lpvt_1a6016c8e736ecef523fbe2539419b5a=1622689578"}

"""
    电子签名数据准备
"""
def get_form_data():
    """
    获取form表单
    :return: 随机的表单数据
    """
    global GLOBAL_NAME
    # 表单数据
    form = {"FontSize" : FONT_SIZE, "FontColor" : FONT_COLOR, "ImageWidth" : IMAGE_WIDTH,
            "ImageHeight" : IMAGE_HEIGHT, "ActionCategory" : ACTION_CATEGORY, "ImageBgColor" : IMAGE_BG_COLOR}
    font_info_id = choice(FONT_INFO_ID)
    form["FontInfoId"] = font_info_id
    content = ""
    while True:
        # 名字的长度
        last_name_count = random.randint(1, 2)
        first_name_index = random.randint(0, len(FIRST_NAME) - 1)
        content += FIRST_NAME[first_name_index]
        for i in range(last_name_count):
            last_name_index = random.randint(0, len(LAST_NAME) - 1)
            content += LAST_NAME[last_name_index]
        if content not in GLOBAL_NAME:
            GLOBAL_NAME.add(content)
            break
        else:
            content = ""
    form["Content"] = content
    return form

def main():
    while len(GLOBAL_NAME) < 400:
        form = get_form_data()
        response = requests.post(url="http://www.diyiziti.com/qianming", data=form, headers=HEADERS)
        content = response.content
        name = form.get("Content")
        with open(r"../tmp/{}.PNG".format(name),"wb") as f:
            f.write(content)
        time.sleep(0.01)

if __name__ == "__main__":
    main()
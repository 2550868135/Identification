import cv2
import os
import numpy as np
from PIL import Image


def read_id_card_path(file_pathname):
    # 遍历该目录下的所有图片文件
    i = 1
    for filename in os.listdir(file_pathname):
        print(filename)
        suffix = filename.split(".")[1]

        # 读入图片保存到img
        img = cv2.imread(file_pathname + '/' + filename)
        # 改变图像大小
        print('Original Dimensions : ', img.shape)
        # scale_percent = 60  # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)

        width = 512
        height = 512
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        print('Resized Dimensions : ', resized.shape)
        #####save figure
        cv2.imwrite("./picture" + "/" + str(i) + '.' + suffix, resized)
        i = i + 1

def PNG_JPG(PngPath):
    img = cv2.imdecode(np.fromfile(os.path.join(PngPath,"蔡昊劲.png"), dtype=np.uint8),cv2.IMREAD_UNCHANGED)
    h, w, _ = img.shape
    infile = PngPath
    outfile = os.path.splitext(infile)[0] + ".jpg"
    img = Image.open(infile)
    try:
        if len(img.split()) == 4:
            # prevent IOError: cannot write mode RGBA as BMP
            r, g, b, a = img.split()
            img = Image.merge("RGB", (r, g, b))
            img.convert('RGB').save(outfile, quality=100)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=100)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


def read_signature_path(file_pathname):
    for filename in os.listdir(file_pathname):
        # 修改文件后缀名为jpg
        os.rename(file_pathname+"/"+filename,file_pathname+"/"+filename[:-4]+".png")
        print(filename)
        # img = cv2.imdecode(np.fromfile(os.path.join(file_pathname, filename), dtype=np.uint8), -1)
        # print('Original Dimensions : ', img.shape)
        # # 改变图像大小
        # # 在上张图片的基础上，上下各填充125像素，填充值为128，生成新的的图像
        # pad_img = cv2.copyMakeBorder(img, 225, 225, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        # # pad_img[:,:,3]=np.ones([570,570])*1
        # print('padded Dimensions : ', pad_img.shape)
        # # 裁剪
        # crop_img = pad_img[29:541, 29:541]
        # print('cropped Dimensions : ', crop_img.shape)
        # cv2.imencode('.jpg', img)[1].tofile("./picture/name/" + filename[:-4] + ".jpg")


# 读取的目录
# read_id_card_path(r"G:\workspace3\identification\data\id_card")
# read_signature_path(r"G:\workspace3\identification\data\name")
PNG_JPG(r"G:\workspace3\identification\data\name")
print(os.getcwd())

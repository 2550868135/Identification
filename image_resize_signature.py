import cv2
import numpy as np
import os
from PIL import Image


def PNG_JPG(PngPath):
    img = cv2.imdecode(np.fromfile(os.path.join(PngPath, "蔡昊劲.png"), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
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
            cv2.imshow("img", img)
            os.remove(PngPath)
        else:
            img.convert('RGB').save(outfile, quality=100)
            os.remove(PngPath)
        return outfile
    except Exception as e:
        print("PNG转换JPG 错误", e)


img_origin = cv2.imread("./1.jpg", 1)
img = cv2.resize(img_origin,[512,512], interpolation=cv2.INTER_AREA)
im = Image.fromarray(img).transpose(Image.ROTATE_90)
img = np.array(im)
cv2.imshow("im", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



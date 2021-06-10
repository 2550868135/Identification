import os

# 旋转后的总图像数
import random

import cv2

MAX_NUM = 200
# 角度边界
ANGLE_EDGE = 330

def rotate_image(base_dir, file, angle):
    """
    旋转图像并保存
    :param angle: 要旋转的角度
    :param base_dir: 要保存的目录
    :param file: 当前图像
    :return:
    """
    name, suffix = os.path.splitext(file)
    filepath = os.path.join(base_dir, file)
    filename = "{name}-{angle}{suffix}".format(name = name,angle = angle, suffix = suffix)
    # 保存路径
    try:
        save_path = os.path.join(base_dir, filename)

        img = cv2.imread(filepath, 1)
        rows, cols, _ = img.shape
        # 旋转中心x,旋转中心y，旋转角度，缩放因子
        M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 0.6)
        # 在内存里完成了旋转
        rotated_img = cv2.warpAffine(
            img, M, (cols, rows),
            borderValue=(127,127,127)
        )
        cv2.imwrite(save_path, rotated_img)
        print("generate image: {}".format(filename))
    except Exception as e:
        pass


def generate_image(path):
    """
    生成训练数据,使样本更均匀
    :param path: 待旋转的数据集的路径
    :return:
    """
    files = os.listdir(path)
    # 文件夹下的文件数
    file_nums = len(files)
    # 每张图片旋转的次数
    base_count = MAX_NUM // file_nums - 1
    # 剩余的数量
    remain_count = MAX_NUM % file_nums
    # 每张图片要旋转的次数
    count_list = [base_count for i in range(file_nums)]
    for i in range(remain_count):
        count_list[i] += 1
    # 遍历文件列表,进行指定次数的旋转
    for index, file in enumerate(files):
        last_angle = 0
        rotate_count = count_list[index]
        for i in range(rotate_count):
            # 随机旋转角度需要进行选址
            angle = last_angle
            # 防止旋转角度过小或过大
            while (angle - 20) < last_angle or (angle - ANGLE_EDGE//rotate_count) > last_angle:
                angle = random.randint(last_angle, ANGLE_EDGE)
            last_angle = angle
            rotate_image(path, file, angle)

if __name__ == "__main__":

    generate_image("../../data/positive_id_card")
    generate_image("../../data/negative_id_card")
    generate_image("../../data/positive_social_security_card")
    generate_image("../../data/negative_social_security_card")

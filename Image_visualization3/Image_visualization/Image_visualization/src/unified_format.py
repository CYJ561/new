import os
import cv2


class unified_format:

    def __init__(self, path, path2):
        self.path = path
        self.path2 = path2


    def united(self):

        images = os.listdir(self.path)  # 返回指定路径下的文件和文件夹列表。
        for i in images:
            # print(os.path.splitext(i))  # 返回文件名和其后缀组成的元组
            if os.path.splitext(i)[1] == ".jpeg":  # 后缀为.jpeg
                img = cv2.imread(self.path + i)  # 读取指定文件图像
                # print(img)
                new_imageName = i.replace(".jpeg", ".png")  # 修改文件后缀
                cv2.imwrite(self.path2 + new_imageName, img)  # 修改后的图像保存到path2中

            elif os.path.splitext(i)[1] == ".jpg":  # 后缀为.jpg
                img = cv2.imread(self.path + i)
                # print(img)
                new_imageName = i.replace(".jpg", ".png")
                cv2.imwrite(self.path2 + new_imageName, img)

            elif os.path.splitext(i)[1] == ".JPG":  # 后缀为.JPG
                img = cv2.imread(self.path + i)
                # print(img)
                new_imageName = i.replace(".JPG", ".png")
                cv2.imwrite(self.path2 + new_imageName, img)

            elif os.path.splitext(i)[1] == ".PNG":  # 后缀为.PNG
                img = cv2.imread(self.path + i)
                # print(img)
                new_imageName = i.replace(".PNG", ".png")
                cv2.imwrite(self.path2 + new_imageName, img)

            elif os.path.splitext(i)[1] == ".png":  # 后缀为.png
                img = cv2.imread(self.path + i)
                # print(img)
                cv2.imwrite(self.path2 + i, img)



"""
Image.new( mode, size, color )创建一幅给定模式（mode）和尺寸（size）的图片。
如果省略 color 参数，则创建的图片被黑色填充满
如果 color 参数是 None 值，则图片还没初始化。

zip() 函数
用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

img.resize((width, height),Image.ANTIALIAS) 该表图片大小
Image.NEAREST ：低质量
Image.BILINEAR：双线性
Image.BICUBIC ：三次样条插值
Image.ANTIALIAS：高质量

image.paste(b,(x,y))：在image的位置（x，y）处将b图像贴上去。

"""



import os
from PIL import Image
# import matplotlib.pyplot as PLT
import numpy as np
# from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# import matplotlib.image as read_png
# from matplotlib.artist import Artist
from sklearn.manifold import TSNE


img_feat = None  # 需要替换成输入的特征
tsne = TSNE(n_components=2, random_state=0, perplexity=10, learning_rate=1500)
res = tsne.fit_transform(img_feat)
tx, ty = res[:,  0], res[:, 1]
tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))

bk_width = 4000
bk_height = 3000
max_dim = 200

# 创建一幅给定模式（mode）和尺寸（size）的图片
back = Image.new('RGB', (bk_width, bk_height), (255, 255, 255))


for id, x, y in zip(modelid_unique, tx, ty):
    img_path = "/home/wangye/Joint-CLIP/Data/tricolo/datasets/224"
    if os.path.exists(os.path.join(img_path, '03001627/'+str(id))):  # os.path.exists()判断括号里的文件是否存在的意思，括号内的可以是文件路径
        img_path = os.path.join(img_path, '03001627/'+str(id)+'/'+'0.png')  # os.path.join()用于路径拼接文件路径，可以传入多个路径
    else:
        img_path = os.path.join(img_path, '04379243/'+str(id)+'/'+'0.png')

    img = Image.open(img_path)  # 图像读取
    rs = max(1, img.width/max_dim, img.height/max_dim)  # 比较选出最大值
    img = img.resize((int(img.width/rs), int(img.height/rs)), Image.ANTIALIAS)  # 该表图片大小
    back.paste(img, (int((bk_width-max_dim)*x), int((bk_height-max_dim)*y)), img)

back.save('res_table.png')

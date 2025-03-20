import time
import tkinter as tk
import os
# 自定义：文件读写IO相关
from fio import *
from PIL import Image, ImageTk

# 自定义: 库的导入管理
# from pio import *


# 初始化窗口宽度
S_WIDTH = 876
# 初始化窗口高度
S_HEIGHT = 700
# 初始化图片最大宽度
I_WIDTH = 710
# 初始化图片最大高度
I_HEIGHT = 670


# 获取当前文件夹的所有图片信息
def get_caches():
    current_dir = os.getcwd()
    cache_dir = current_dir + '/' + IMAGE_CACHE_DIR + '/'
    caches = os.listdir(cache_dir)
    temp_caches = []
    for cache in caches:
        temp_caches.append(cache_dir + cache)
    caches.sort(key=lambda fn: os.path.getmtime(cache_dir + fn), reverse=True)
    temp_caches.sort(key=lambda fn: os.path.getmtime(fn), reverse=True)
    return caches, temp_caches


# 按比例缩放图片
def resize(path, scale=-1):
    image = Image.open(path)
    # 按照当前窗口的大小最大缩放图片
    if scale == -1:
        raw_width, raw_height = image.size[0], image.size[1]
        max_width, max_height = I_WIDTH, I_HEIGHT
        min_height = min(max_height, raw_height)
        min_width = int(raw_width * min_height / raw_height)
        if min_width > max_width:
            min_width = min(max_width, raw_width)
            min_height = int(raw_height * min_width / raw_width)
    # 缩略图缩放
    elif scale == 1:
        raw_width, raw_height = image.size[0], image.size[1]
        min_height = 166
        min_width = int(raw_width * min_height / raw_height)
    # 工具栏图标icon缩放
    else:
        raw_width, raw_height = image.size[0], image.size[1]
        min_height = 20
        min_width = int(raw_width * min_height / raw_height)
    return image.resize((min_width, min_height))


class MyGallery:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title('ImageCape')
        self.window.wm_title('ImageCape')
        self.window.geometry(str(S_WIDTH) + 'x' + str(S_HEIGHT) + '+50+200')

        self.frame = tk.Frame(self.window)
        # self.frame2 = tk.Frame(self.window)
        self.frame3 = tk.Frame(self.window)
        self.window.config(bg='#111')

        self.window.rowconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        # self.window.rowconfigure(2, weight=1)
        self.window.columnconfigure(0, weight=1)
        self.window.columnconfigure(1, weight=1)
        # self.window.columnconfigure(2, weight=1)

        self.frame.columnconfigure(0, weight=1)
        self.frame3.rowconfigure(0, weight=1)
        self.frame3.columnconfigure(0, weight=1)
        self.frame3.config(bg='#111')

        self.remote_url = tk.StringVar()
        self.text = tk.Entry(self.frame, width=S_WIDTH, textvariable=self.remote_url)
        self.text.config(bg='#eee')
        self.text.config(fg='#111')
        self.text.config(highlightcolor='#999')
        self.text.grid(row=0, sticky=tk.E)
        self.text.bind('<Button-1>', self.reset_tip_bar)
        self.text.bind('<Return>', self.load_image_remote)

        # 下载图片按钮，略
        self.enter_button = tk.Button(self.frame, text='Cap it!', width=20, height=20)
        go_icon = ImageTk.PhotoImage(resize(os.getcwd() + '/go.jpeg', 0))
        self.enter_button.config(image=go_icon)
        self.enter_button.bind('<Button-1>', self.load_image_remote)
        self.enter_button.grid(row=0, column=1, sticky=tk.E)
        # 删除按钮
        self.del_button = tk.Button(self.frame, text='DEL', width=20, height=20)
        # Bug occurs in the following code.
        # self.del_button.config(image=ImageTk.PhotoImage(resize(os.getcwd() + '/delete.png', 0)))
        del_icon = ImageTk.PhotoImage(resize(os.getcwd() + '/delete.png', 0))
        self.del_button.config(image=del_icon)
        self.del_button.bind('<Button-1>', self.delete_selected_image)
        self.del_button.grid(row=0, column=2, sticky=tk.W)
        # 地址栏和工具栏的布局
        self.frame.grid(row=0, sticky=tk.N + tk.W + tk.E)
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)

        # self.frame2.grid(row=2, column=0, sticky=tk.W+tk.E)
        # self.frame2.rowconfigure(0, weight=1)
        # self.frame2.columnconfigure(1, weight=1)
        # self.frame2.columnconfigure(0, weight=1)

        self.caches, self.cache_paths = get_caches()
        self.image_pos = 0
        if len(self.caches) > 0 and len(self.caches) > self.image_pos >= 0:
            image = resize(self.cache_paths[self.image_pos])
            photo = ImageTk.PhotoImage(image)
        else:
            photo = None
        # 显示当前图片
        self.label = tk.Label(self.frame3, text=self.caches[self.image_pos],
                              padx=1, pady=1, image=photo)
        self.label.config(bg='#111')
        self.label.config(justify='center')
        self.label.focus_get()
        # 点击鼠标左键，显示下一张图片的大图
        self.label.bind('<Button-1>', self.set_next_image)
        # 点击鼠标右键，则显示上一张大图
        self.label.bind('<Button-3>', self.set_prev_image)
        self.label.grid(row=0, column=0, rowspan=4, sticky=tk.NSEW)
        # 缩略图列表
        self.sub_label = tk.Label(self.frame3, text='Empty', width=166, height=166,
                                  padx=1, pady=1)
        self.sub_label.config(bg='#111')
        self.sub_image = None
        self.sub_photo = None
        # 绑定第1张缩略图的鼠标左键点击事件为self.load_sub_image_1——显示缩略图的大图
        self.sub_label.bind('<Button-1>', self.load_sub_image_1)
        self.sub_label.grid(row=0, column=1, sticky=tk.NE)

        self.sub_label2 = tk.Label(self.frame3, text='Empty', width=166, height=166, padx=1, pady=1)
        self.sub_label2.config(bg='#111')
        self.sub_image2 = None
        self.sub_photo2 = None
        # 绑定第2张缩略图的鼠标左键点击事件为self.load_sub_image_2——显示缩略图的大图
        self.sub_label2.bind('<Button-1>', self.load_sub_image_2)
        self.sub_label2.grid(row=1, column=1, sticky=tk.NE)

        self.sub_label3 = tk.Label(self.frame3, text='Empty', width=166, height=166, padx=1, pady=1)
        self.sub_label3.config(bg='#111')
        self.sub_image3 = None
        self.sub_photo3 = None
        # 绑定第3张缩略图的鼠标左键点击事件为self.load_sub_image_3——显示缩略图的大图
        self.sub_label3.bind('<Button-1>', self.load_sub_image_3)
        self.sub_label3.grid(row=2, column=1, sticky=tk.NE)

        self.sub_label4 = tk.Label(self.frame3, text='Empty', width=166, height=166, padx=1, pady=1)
        self.sub_label4.config(bg='#111')
        self.sub_image4 = None
        self.sub_photo4 = None
        # 绑定第4张缩略图的鼠标左键点击事件为self.load_sub_image_4——显示缩略图的大图
        self.sub_label4.bind('<Button-1>', self.load_sub_image_4)
        self.sub_label4.grid(row=3, column=1, sticky=tk.NE)

        self.reset_sub_image()
        # 大图和缩略图列表的UI界面布局
        self.frame3.grid(row=1, column=0, sticky=tk.N + tk.W + tk.E)
        self.frame3.columnconfigure(0, weight=1)
        self.frame3.rowconfigure(0, weight=1)
        # 上一张图片按钮
        self.prev_button = tk.Button(self.frame, width=20, height=20, text='Prev')
        prev_icon = ImageTk.PhotoImage(resize(os.getcwd() + '/backward.png', 0))
        self.prev_button.config(image=prev_icon)
        self.prev_button.config(bd=1)
        # 绑定上一张按钮的鼠标左键点击事件
        self.prev_button.bind('<Button-1>', self.set_prev_image)
        self.prev_button.grid(row=0, column=3, padx=0, sticky=tk.E)
        # 下一张图片按钮
        self.next_button = tk.Button(self.frame, width=20, height=20, text='Next')
        next_icon = ImageTk.PhotoImage(resize(os.getcwd() + '/forward.png', 0))
        self.next_button.config(image=next_icon)
        # 绑定下一张按钮的鼠标左键点击事件
        self.next_button.bind('<Button-1>', self.set_next_image)
        self.next_button.grid(row=0, column=4, padx=0, sticky=tk.E)

        self.window.mainloop()

    def __del__(self):
        self.window = None

    def reset_tip_bar(self, event):
        self.text.config(highlightcolor='#999')

    def reload_caches(self):
        self.caches, self.cache_paths = get_caches()

    # 重置缩略图列表
    def reset_sub_image(self):
        # sub image 1
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 1 > 0:
            self.sub_image = resize(self.cache_paths[self.image_pos + 1], 1)
            self.sub_photo = ImageTk.PhotoImage(self.sub_image)
        elif len(self.caches) > 0 and len(self.caches) == self.image_pos + 1:
            self.sub_image = resize(self.cache_paths[len(self.caches) - 1], 1)
            self.sub_photo = ImageTk.PhotoImage(self.sub_image)
        else:
            self.sub_photo = None
        self.sub_label.config(image=self.sub_photo)
        # sub image 2
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 2 > 0:
            self.sub_image2 = resize(self.cache_paths[self.image_pos + 2], 1)
            self.sub_photo2 = ImageTk.PhotoImage(self.sub_image2)
        else:
            self.sub_photo2 = None
        self.sub_label2.config(image=self.sub_photo2)
        # sub image 3
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 3 > 0:
            self.sub_image3 = resize(self.cache_paths[self.image_pos + 3], 1)
            self.sub_photo3 = ImageTk.PhotoImage(self.sub_image3)
        else:
            self.sub_photo3 = None
        self.sub_label3.config(image=self.sub_photo3)
        # sub image 4
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 4 > 0:
            self.sub_image4 = resize(self.cache_paths[self.image_pos + 4], 1)
            self.sub_photo4 = ImageTk.PhotoImage(self.sub_image4)
        else:
            self.sub_photo4 = None
        self.sub_label4.config(image=self.sub_photo4)

    # 加载第1张缩略图
    def load_sub_image_1(self, event):
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 1 > 0:
            self.image_pos += 1
            self.load_image(self.image_pos)
        self.reset_sub_image()

    # 加载第2张缩略图
    def load_sub_image_2(self, event):
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 2 > 0:
            self.image_pos += 2
            self.load_image(self.image_pos)
        self.reset_sub_image()

    # 加载第3张缩略图
    def load_sub_image_3(self, event):
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 3 > 0:
            self.image_pos += 3
            self.load_image(self.image_pos)
        self.reset_sub_image()

    # 加载第4张缩略图
    def load_sub_image_4(self, event):
        if len(self.caches) > 0 and len(self.caches) > self.image_pos + 4 > 0:
            self.image_pos += 4
            self.load_image(self.image_pos)
        self.reset_sub_image()

    # 图片下载器ImgLoader，此处略
    def load_image_remote(self, event):
        url = self.remote_url.get()
        self.text.delete(0, tk.END)
        # 简单验证URL地址是否有效
        if not verify_path_loose(url):
            return
        print('[process]', 'forked')
        start_time = time.time()
        time.sleep(.1)
        # 下载图片流程略
        ...
        end_time = time.time()
        print('[process]', 'done with', '%.3f' % (end_time - start_time) + ' s.')

    # 删除当前显示的图片
    def delete_selected_image(self, event=None):
        if len(self.caches) > 0 and len(self.caches) > self.image_pos >= 0:
            try:
                os.remove(self.cache_paths[self.image_pos])
                self.text.config(highlightcolor='yellow')
                self.reload_caches()
                self.set_prev_image()
            except FileNotFoundError:
                self.text.config(highlightcolor='red')

    # 显示当前定位的图片
    def load_image(self, position=0):
        if len(self.caches) > 0 and len(self.caches) > position >= 0:
            try:
                image = resize(self.cache_paths[position])
                photo = ImageTk.PhotoImage(image)
                self.label.config(image=photo)
                self.label.image = photo
                # print(self.caches[position])
            except FileNotFoundError:
                self.reload_caches()
        else:
            photo = None
            self.label.config(image=photo)
            self.label.image = photo

    # 上一张图片
    def set_prev_image(self, event=None):
        if len(self.caches) > 0 and len(self.caches) > self.image_pos > 0:
            self.image_pos -= 1
        else:
            self.reload_caches()
            self.image_pos = 0
        self.load_image(self.image_pos)
        # 更新缩略图列表
        self.reset_sub_image()

    # 下一张图片
    def set_next_image(self, event=None):
        if len(self.caches) > 0 and len(self.caches) - 1 > self.image_pos >= 0:
            self.image_pos += 1
        else:
            self.reload_caches()
            self.image_pos = len(self.caches) - 1
        self.load_image(self.image_pos)
        # 更新缩略图列表
        self.reset_sub_image()


my_gallery = MyGallery()
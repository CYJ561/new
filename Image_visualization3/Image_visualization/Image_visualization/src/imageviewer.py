import tkinter as tk
from tkinter.filedialog import askdirectory
from src2.imageutil import *
from PIL import ImageTk
import os

DEFAULT_IMAGE_DIR = 'Destination'


# 检索或创建文件目录
def init_dir():
    img_dir = os.getcwd() + '/' + DEFAULT_IMAGE_DIR  # 返回当前工作目录+/img

    # 若文件目录不存在
    if not os.path.exists(img_dir):
        # 创建一个目录img_dir
        os.makedirs(img_dir)


# 对工作目录中文件后缀修改并返回文件
def load_cache(media_dir=None):
    # 工作目录为空
    if media_dir is None:
        current_dir = os.path.abspath('..') + '/' + DEFAULT_IMAGE_DIR  # 工作目录的绝对路径
    else:  # 不为空
        current_dir = media_dir
        if not current_dir.startswith('/'):  # 工作目录是否以'/'开头
            current_dir = os.path.abspath('..') + '/' + current_dir  # 变为绝对路径
    files = os.listdir(current_dir)  # 获取指定路径下的所有文件
    temp = []
    for file in files:
        # 若该路径为文件
        if os.path.isfile(current_dir + '/' + file):  # 绝对路径
            temp.append(current_dir + '/' + file)  # 添加到temp中
    # 路径下无文件则返回空元组
    if len(temp) < 1:
        return []
    files.clear()  # 清空files
    for file in temp:
        # 选择带后缀的文件并去除后缀，改变为小写字符
        if '.' in file and file.split('.')[-1].lower() in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']:
            files.append(file)
    files.sort()
    # print(files)
    return files


class MyWindow:
    BUTTON_SIZE_NORMAL = 32

    def __init__(self):
        self.x = 0
        self.y = 0
        self.cursor = 0
        self.start_pos = 0
        self.first_load = True
        self.buttons = None
        self.photo = None
        self.sub_image = None   # 主窗口1，存放大图
        self.sub_image2 = None  # 小窗2
        self.sub_image3 = None  # 小窗3
        self.sub_image4 = None  # 小窗4
        self.sub_image5 = None  # 小窗5
        self.sub_queue = []
        init_dir()   # 检索工作目录
        # 创建应用程序底层窗口
        self.window = tk.Tk()
        # 更改标题
        self.window.title('imageViewer')
        # 底层窗口配色（黑色）
        self.window.config(bg='#111')
        #  不去掉任务栏
        self.window.overrideredirect(False)
        # 创建Frame容器
        self.frame_left_bar = tk.Frame(self.window)  # 边框容器Frame
        self.frame_photo = tk.Frame(self.window)    # 主窗口1的Frame

        self.window_width = 0
        self.window_height = 0
        # 设置窗口宽和高
        self.window.geometry(
            f'{M_WIDTH}x{M_HEIGHT}+{(self.window.winfo_screenwidth() - S_WIDTH) // 2}'
            f'+{(self.window.winfo_screenheight() - S_HEIGHT) // 2 - 18}')
        # 绑定窗口事件，窗口尺寸发生改变的时候触发该事件
        self.window.bind('<Configure>', self.window_resize)
        # self.window.bind('<B1-Motion>', self.win_motion)
        # 关闭窗口时，触发win_quit事件，销毁窗口
        self.window.protocol('WM_DELETE_WINDOW', self.win_quit)
        # 动态变换窗口大小
        # self.window.bind('<Button-1>', self.left_down)
        # 设置主窗口颜色
        self.frame_left_bar.config(bg='#111')
        # 设置主窗口容器宽
        self.frame_left_bar.config(width=SUB_WIDTH)
        # 设置画布，置于主窗口1容器内
        self.photo_can = tk.Canvas(self.frame_photo)
        self.selected_dir = DEFAULT_IMAGE_DIR  # 工作目录Destination
        # 获得存放工作目录文件的数组
        self.caches = load_cache(self.selected_dir)
        '''
        # 若caches不为空
        if len(self.caches) > 0:
            # 将图像文件转换成 PhotoImage 对象，PhotoImage对象可在控件（Button、Label等）上显示图像。
            self.photo = ImageTk.PhotoImage(image=resize_img(self.caches[self.cursor], screen_width=S_WIDTH, screen_height=S_HEIGHT))
            
            self.photo_can.create_image(((S_WIDTH - self.photo.width()) // 2 - 36, (S_HEIGHT - self.photo.height()) // 2), anchor=tk.NW, image=self.photo)
        else:  # 有东西
            self.photo = None
        '''
        # 设置主窗口画布底色
        self.photo_can.update()
        self.photo_can.config(bg='#111')  # 黑色
        # self.photo_can.config(highlightthickness=0)  # 文本框高亮边框宽度设置
        self.photo_can.config(width=M_WIDTH - M_PAD_X - SUB_WIDTH)  # 设置画布width=560
        self.photo_can.config(height=M_HEIGHT)  # 设置画布height=640

        # 创建向左查看的按钮prev_button，在window上
        self.prev_button = tk.Button(self.window, text='‹')
        # 绑定按钮响应函数prev_photo
        self.prev_button.config(command=self.prev_photo)
        # 设置左键字体大小
        self.prev_button.config(font=('', 16))
        # 设置左键按钮位置
        self.prev_button.place(x=0, y=(S_HEIGHT - self.BUTTON_SIZE_NORMAL) // 2, width=self.BUTTON_SIZE_NORMAL, height=self.BUTTON_SIZE_NORMAL)
        # 设置画布停靠在左，随窗口缩放，在水平和竖直方向填充，组件居中
        self.photo_can.pack(side='left', expand='yes', fill='both', anchor='center')

        # 创建向右查看的按钮next_button，在window上
        self.next_button = tk.Button(self.window, text='›')
        # 绑定按钮响应函数next_photo
        self.next_button.config(command=self.next_photo)
        self.next_button.config(font=('', 16))
        self.next_button.place(x=S_WIDTH - self.BUTTON_SIZE_NORMAL, y=(S_HEIGHT - self.BUTTON_SIZE_NORMAL) // 2, width=self.BUTTON_SIZE_NORMAL, height=self.BUTTON_SIZE_NORMAL)

        # 创建向上翻页的按钮prev_page_button，在window上
        self.prev_page_button = tk.Button(self.window, text='⌃')
        # 绑定按钮响应函数prev_page
        self.prev_page_button.config(command=self.prev_page)
        self.prev_page_button.config(font=('', 10))

        # 创建向下翻页的按钮next_page_button，在window上
        self.next_page_button = tk.Button(self.window, text='⌄')
        self.next_page_button.config(command=self.next_page)
        self.next_page_button.config(font=('', 10))

        # 创建按钮open_button，用于打开另一个文件夹中的照片库
        self.open_button = tk.Button(self.window, text='…')
        self.open_button.config(command=self.select_dir)  # 绑定按钮响应函数
        self.open_button.config(font=('', 10))

        # 创建按钮reload_button，用于返回到初始状态（浏览到后方后需要回到原位置）
        self.reload_button = tk.Button(self.window, text='↺')
        self.reload_button.config(command=self.reload_cache)  # 响应函数
        self.reload_button.config(font=('', 10))

        # 创建按钮delete_button，用于删除正在查看的图片
        self.delete_button = tk.Button(self.window, text='⤫')
        self.delete_button.config(command=self.delete_cache)
        self.delete_button.config(font=('', 10))
        # 配置7个按钮的背景
        self.config_button()

        # 按钮1，边框第一个图片预览组件
        self.sub_image_can = tk.Button(self.window)
        self.sub_image_can.config(command=self.load_sub_1)
        # 按钮2，边框第二个图片预览组件
        self.sub_image_can2 = tk.Button(self.window)
        self.sub_image_can2.config(command=self.load_sub_2)
        # 按钮3，边框第三个图片预览组件
        self.sub_image_can3 = tk.Button(self.window)
        self.sub_image_can3.config(command=self.load_sub_3)
        # 按钮4，边框第四个图片预览组件
        self.sub_image_can4 = tk.Button(self.window)
        self.sub_image_can4.config(command=self.load_sub_4)
        # 按钮5，边框第五个图片预览组件
        self.sub_image_can5 = tk.Button(self.window)
        self.sub_image_can5.config(command=self.load_sub_5)
        # 加载边框的五个缩略图
        self.load_sub()
        # 配置边框按钮背景
        self.sub_image_cans = [self.sub_image_can, self.sub_image_can2, self.sub_image_can3, self.sub_image_can4, self.sub_image_can5]
        self.config_can()
        # 标签pointer，指出正在浏览的图片
        self.pointer = tk.Label(self.window, text='√')
        self.pointer.config(height=1)
        self.pointer.config(bg='#333')
        self.pointer.config(fg='#eee')
        # 设置边框容器停靠在右，随窗口缩放，在竖直方向填充
        self.frame_left_bar.pack(side='right', expand='yes', fill='y')
        # 设置主窗口容器停靠在右，随窗口缩放，在竖直和水平方向填充
        self.frame_photo.pack(side='right', expand='yes', fill='both')

        self.window.mainloop()  # 开始运行并加载

    # 销毁底层窗口，释放内存空间
    def win_quit(self, event=None):
        self.window.quit()
        self.__del__()

    # 窗口最小化
    # def win_mini(self, event=None):
    #     self.window.state('icon')  # 最小化
    #     self.window.iconify()  # 设置窗口最小化
    # 获得事件坐标
    # def left_down(self, event=None):
    #     self.x = event.x
    #     self.y = event.y
    # def win_motion(self, event):
    #     new_x = int(event.x - self.x) + self.window.winfo_x()
    #     new_y = int(event.y - self.y) + self.window.winfo_y()
    #     self.window.geometry(f'{self.window.winfo_width()}x{self.window.winfo_height()}+{new_x}+{new_y}')

    # 设置边框五个图片按钮的背景
    def config_can(self):
        for image_can in self.sub_image_cans:
            image_can.config(relief='ridge')
            image_can.config(fg='#fff')
            image_can.config(activeforeground='#f5f5f5')
            image_can.config(activebackground='#444')
            image_can.config(bg='#222')
            image_can.config(bd=0)
            image_can.config(highlightthickness=0)
            image_can.config(highlightcolor='#111')
            image_can.config(highlightbackground='#111')

    # 设置底层窗口的7个按钮的背景央视
    def config_button(self):
        self.buttons = [self.prev_button, self.next_button, self.prev_page_button, self.next_page_button,
                        self.delete_button, self.open_button, self.reload_button]
        for button in self.buttons:
            button.config(relief='ridge')
            button.config(fg='#fff')
            button.config(activeforeground='#f5f5f5')
            button.config(activebackground='#444')
            button.config(bg='#222')
            button.config(bd=0)
            button.config(highlightthickness=0)
            button.config(highlightcolor='#111')
            button.config(highlightbackground='#111')

    # 转换图像文件格式为Image后返回
    def get_sub_image(self, file=None):

        # 将图像文件转换成PhotoImage对象，PhotoImage对象可在控件（Button、Label等）上显示图像。
        return ImageTk.PhotoImage(image=resize_img(file, 1, screen_width=M_WIDTH, screen_height=M_HEIGHT))  # 返回图像

    def load_photo(self):
        if len(self.caches) > 0:
            if 0 <= self.cursor <= len(self.caches) - 1:
                width = self.photo_can.winfo_width() if self.photo_can.winfo_width() > 0 else S_WIDTH
                height = self.photo_can.winfo_height() if self.photo_can.winfo_height() > 0 else S_HEIGHT
                image = resize_img(self.caches[self.cursor], screen_width=width, screen_height=height)
                self.photo = ImageTk.PhotoImage(image=image)
                self.photo_can.create_image(
                    ((width - self.photo.width()) // 2,
                     (height - self.photo.height()) // 2),
                    anchor=tk.NW, image=self.photo)
                self.window.title(f'{self.caches[self.cursor].split("/")[-1]}')
        else:
            self.photo = None
        self.photo_can.update()

    def load_sub(self):
        if 0 <= self.start_pos < len(self.caches):
            self.sub_image = self.get_sub_image(file=self.caches[self.start_pos])
        else:
            self.sub_image = self.get_sub_image(file=os.getcwd() + '/bg_empty.png')
        self.sub_image_can.config(image=self.sub_image)
        self.sub_image_can.update()
        if 0 <= self.start_pos + 1 < len(self.caches):
            self.sub_image2 = self.get_sub_image(file=self.caches[self.start_pos + 1])
        else:
            self.sub_image2 = self.get_sub_image(file=os.getcwd() + '/bg_empty.png')
        self.sub_image_can2.config(image=self.sub_image2)
        self.sub_image_can2.update()
        if 0 <= self.start_pos + 2 < len(self.caches):
            self.sub_image3 = self.get_sub_image(file=self.caches[self.start_pos + 2])
        else:
            self.sub_image3 = self.get_sub_image(file=os.getcwd() + '/bg_empty.png')
        self.sub_image_can3.config(image=self.sub_image3)
        self.sub_image_can3.update()
        if 0 <= self.start_pos + 3 < len(self.caches):
            self.sub_image4 = self.get_sub_image(file=self.caches[self.start_pos + 3])
        else:
            self.sub_image4 = self.get_sub_image(file=os.getcwd() + '/bg_empty.png')
        self.sub_image_can4.config(image=self.sub_image4)
        self.sub_image_can4.update()
        if 0 <= self.start_pos + 4 < len(self.caches):
            self.sub_image5 = self.get_sub_image(file=self.caches[self.start_pos + 4])
        else:
            self.sub_image5 = self.get_sub_image(file=os.getcwd() + '/bg_empty.png')
        self.sub_image_can5.config(image=self.sub_image5)
        self.sub_image_can5.update()

    def load_sub_1(self):
        if len(self.caches) > self.start_pos:
            self.cursor = self.start_pos
            self.load_photo()
            self.point_to()

    def load_sub_2(self):
        if len(self.caches) > self.start_pos + 1:
            self.cursor = self.start_pos + 1
            self.load_photo()
            self.point_to()

    def load_sub_3(self):
        if len(self.caches) > self.start_pos + 2:
            self.cursor = self.start_pos + 2
            self.load_photo()
            self.point_to()

    def load_sub_4(self):
        if len(self.caches) > self.start_pos + 3:
            self.cursor = self.start_pos + 3
            self.load_photo()
            self.point_to()

    def load_sub_5(self):
        if len(self.caches) > self.start_pos + 4:
            self.cursor = self.start_pos + 4
            self.load_photo()
            self.point_to()

    def point_to(self):
        delta_cursor = self.cursor - self.start_pos
        if 0 <= delta_cursor < len(self.sub_image_cans):
            width = SUB_WIDTH // 6
            height = SUB_HEIGHT
            x = self.window.winfo_width() - SUB_WIDTH
            y = self.sub_image_can.winfo_y() + delta_cursor * height
            if self.sub_image_can.winfo_y() < M_PAD_Y:
                y += M_PAD_Y
            self.pointer.place(x=x, y=y, width=width, height=height)

    def prev_photo(self):
        # 若工作目录不为空
        if len(self.caches) > 0:
            updated = False  # 更新状态初始化
            self.cursor -= 1  # 初始为-1
            if self.cursor < self.start_pos:  # -1<0
                updated = True  # 可以修改
                if self.start_pos - 5 >= 0:
                    self.start_pos -= 5
                else:
                    self.start_pos = len(self.caches) - 5
            if self.cursor < 0:
                updated = True
                self.cursor = len(self.caches) - 1
            self.load_photo()
            if updated:
                self.load_sub()
            self.point_to()

    def next_photo(self):
        if len(self.caches) > 0:
            updated = False
            self.cursor += 1
            if self.cursor > self.start_pos + 4:
                updated = True
                if self.start_pos + 5 < len(self.caches):
                    self.start_pos += 5
                else:
                    self.start_pos = 0
            if self.cursor >= len(self.caches):
                updated = True
                self.cursor = 0
                self.start_pos = 0
            self.load_photo()
            if updated:
                self.load_sub()
            self.point_to()

    def prev_page(self, event=None):
        if self.start_pos - 5 >= 0:
            self.start_pos -= 5
        else:
            self.start_pos = len(self.caches) - 5
        self.cursor = self.start_pos + 4
        self.load_photo()
        self.load_sub()
        self.point_to()

    def next_page(self, event=None):
        if self.start_pos + 5 < len(self.caches):
            self.start_pos += 5
        else:
            self.start_pos = 0
        self.cursor = self.start_pos
        self.load_photo()
        self.load_sub()
        self.point_to()

    def window_resize(self, event=None):
        # 窗口尺寸发生变化
        if event is not None:
            # listen events of window resizing.
            if self.window_width != self.photo_can.winfo_width() or self.window_height != self.photo_can.winfo_height():
                if self.window_width != self.photo_can.winfo_width():
                    self.window_width = self.photo_can.winfo_width()
                if self.window_height != self.photo_can.winfo_height():
                    self.window_height = self.photo_can.winfo_height()
                # What happens here?
                if self.first_load:
                    self.first_load = False
                else:
                    self.photo_can.config(width=self.window.winfo_width() - M_PAD_X - SUB_WIDTH)
                    self.photo_can.config(height=self.window.winfo_height())
                    self.photo_can.update()
                    self.load_photo()
                    self.place_page_tool()
                    self.place_tool()
                    self.prev_button.place(
                        x=0,
                        y=(self.window_height - self.BUTTON_SIZE_NORMAL) // 2,
                        width=self.BUTTON_SIZE_NORMAL,
                        height=self.BUTTON_SIZE_NORMAL)
                    self.next_button.place(
                        x=self.window_width - self.BUTTON_SIZE_NORMAL,
                        y=(self.window_height - self.BUTTON_SIZE_NORMAL) // 2,
                        width=self.BUTTON_SIZE_NORMAL,
                        height=self.BUTTON_SIZE_NORMAL)
                    self.place_sub()
                    self.point_to()

    def place_page_tool(self):
        x = self.window.winfo_width() - SUB_WIDTH
        y = 0
        width = SUB_WIDTH
        height = M_PAD_Y
        self.prev_page_button.place(x=x, y=y, width=width, height=height)
        self.next_page_button.place(x=x, y=self.window_height - M_PAD_Y, width=width, height=height)

    def place_tool(self):
        i = 0
        for button in self.buttons[4:]:
            button.place(
                x=self.window.winfo_width() - SUB_WIDTH - 1 - self.BUTTON_SIZE_NORMAL * (len(self.buttons) - 2 - i),
                y=0,
                width=self.BUTTON_SIZE_NORMAL,
                height=self.BUTTON_SIZE_NORMAL)
            i += 1

    def place_sub(self):
        i = 0
        start_y = (self.window.winfo_height() - SUB_HEIGHT * 5) // 2
        # start_y += self.BUTTON_SIZE_NORMAL
        for sub_can in self.sub_image_cans:
            sub_can.place(x=self.window.winfo_width() - SUB_WIDTH, y=start_y + SUB_HEIGHT * i, width=SUB_WIDTH,
                          height=SUB_HEIGHT)
            i += 1

    # 重置，返回到最原来位置
    def reload_cache(self):
        caches = load_cache(self.selected_dir)
        if len(caches) > 0:
            self.caches = caches
            self.cursor = 0
            self.start_pos = 0
            self.load_sub()
            self.point_to()
            self.load_photo()

    # 打开另一个文件夹中的照片库
    def select_dir(self):
        # Select the directory of photo(s).
        selected_dir = askdirectory()
        # Get () as return if selection is canceled or file dialog close.
        if selected_dir is not None and selected_dir != ():
            self.selected_dir = selected_dir
            caches = load_cache(self.selected_dir)
            if len(caches) > 0:
                self.caches = caches
                self.cursor = 0
                self.start_pos = 0
                self.load_sub()
                self.point_to()
                self.load_photo()

    # 删除
    def delete_cache(self):
        if 0 <= self.cursor <= len(self.caches) - 1:
            cache = self.caches[self.cursor]
            if os.path.exists(cache):
                os.remove(cache)
            self.caches.remove(cache)
            if 0 <= self.cursor <= len(self.caches) - 1:
                pass
            else:
                if len(self.caches) > 0:
                    self.cursor = len(self.caches) - 1
                else:
                    self.cursor = 0
            self.point_to()
            self.load_photo()

    def __del__(self):
        self.photo = None
        self.caches = None
        self.window = None
        del self.photo
        del self.caches
        del self.window


def test():
    my_window = MyWindow()


if __name__ == '__main__':
    test()

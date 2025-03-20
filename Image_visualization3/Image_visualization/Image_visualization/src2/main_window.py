import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image


class main_window:

    def __init__(self):  # 初始化界面

        # 应用程序主窗口main_frame
        self.main_frame = tk.Tk()
        self.main_frame.title('ImageViewer')
        # 设置尺寸
        screen_width = self.main_frame.winfo_screenwidth()
        screen_height = self.main_frame.winfo_screenheight()
        width, height = 1000, 650
        gm_str = "%dx%d+%d+%d" % (width, height, (screen_width - width) / 2,
                                  (screen_height - height) / 2)
        self.main_frame.geometry(gm_str)
        # 去掉边框
        # self.main_frame.overrideredirect(-1)
        #  允许更改窗口大小
        self.main_frame.resizable(width=True, height=True)
        # 设置黑色
        self.main_frame.config(bg='#111')
        # 允许窗口尺寸变化时跟随变化
        # 行跟随
        self.main_frame.rowconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        # 列跟随
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)

        #  创建容器组件，用于存放其他控件
        self.window = Frame(self.main_frame)
        self.window.config(bg='#111')  # 设置黑色


        self.window2 = Frame(self.main_frame)

        # 创建text
        # self.

        # # 画布组件
        # self.canvas = Canvas(self.main_frame, width=1000, height=600, bg="white")
        # self.canvas.pack(anchor="n")
        # # 画线
        # # line = self.canvas.create_line(10, 10, 30, 20, 40, 50)
        # # 图
        # image_file = Image.open("F:\\PythonPro\\Image_visualization\\Destination\\171201bmhbzyxrd944xduj.png")  # 打开图片文件
        # image = image_file.resize((100, 200), Image.ANTIALIAS)
        # tk_image = ImageTk.PhotoImage(image)  # 图片转换为tkinter可用格式
        #
        # self.canvas.create_image(150, 170, image=tk_image)
        #
        # # 2 self.label = Label(self.canvas,image=tk_image).place(x=50,y=100)


        self.main_frame.mainloop()



main = main_window()
main.show()
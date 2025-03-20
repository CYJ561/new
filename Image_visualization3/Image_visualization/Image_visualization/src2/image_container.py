import tkinter as tk

class image_container:

    def __init__(self):  # 初始化界面
        self.x = 0
        self.y = 0
        # 创建应用程序底层窗口
        self.window = tk.Tk()
        self.window.title('image_container')
        # # 画布组件
        self.canvas = tk.Canvas(self.window, width=1000, height=600, bg="white")
        self.canvas.pack(anchor="n")

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

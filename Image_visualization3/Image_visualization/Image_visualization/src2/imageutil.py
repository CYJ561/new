from PIL import Image, ImageTk

S_WIDTH = 560
S_HEIGHT = 640
M_PAD_X = 1
M_PAD_Y = 20
SUB_HEIGHT = (S_HEIGHT - 2 * M_PAD_Y) // 5  # 取整，为120
SUB_WIDTH = SUB_HEIGHT  # 120
M_WIDTH = S_WIDTH + SUB_WIDTH + M_PAD_X  # 681
M_HEIGHT = S_HEIGHT  # 640
MIN_SUB_WIDTH = 16
MIN_SUB_HEIGHT = 16
I_WIDTH = S_WIDTH  # 560
I_HEIGHT = S_HEIGHT  # 640


def resize_img(path, scale=-1, screen_width=0, screen_height=0):
    image = Image.open(path)
    if scale == -1:
        if screen_width <= 0:
            screen_width = I_WIDTH
        if screen_height <= 0:
            screen_height = I_HEIGHT
        raw_width, raw_height = image.size[0], image.size[1]
        # max_width, max_height = I_WIDTH, I_HEIGHT
        max_width, max_height = raw_width, screen_height
        # '''
        min_width = max(raw_width, max_width)
        min_height = int(raw_height * min_width / raw_width)
        while min_height > screen_height:
            min_height = int(min_height * .9533)
        while min_height < screen_height:
            min_height += 1
        min_width = int(raw_width * min_height / raw_height)
        '''
        min_height = max(raw_width, max_width)
        min_width = int(raw_width * min_height / raw_height)
        '''
        while min_width > screen_width:
            min_width -= 1
        min_height = int(raw_height * min_width / raw_width)
    elif scale == 1:
        raw_width, raw_height = image.size[0], image.size[1]
        max_width, max_height = raw_width, SUB_HEIGHT
        # '''
        min_width = max(raw_width, max_width)
        min_height = int(raw_height * min_width / raw_width)
        while min_height > SUB_HEIGHT:
            min_height = int(min_height * .9533)
        while min_height < SUB_HEIGHT:
            min_height += 1
        min_width = int(raw_width * min_height / raw_height)
        while min_width > SUB_WIDTH:
            min_width -= 1
        min_height = int(raw_height * min_width / raw_width)
    else:
        raw_width, raw_height = image.size[0], image.size[1]
        min_height = 18
        min_width = int(raw_width * min_height / raw_height)
    return image.resize((min_width, min_height))


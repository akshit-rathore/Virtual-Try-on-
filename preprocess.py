import numpy as np
import requests
import os
from PIL import Image
import numpy as np
from rembg import remove





class preprocessInput:

    def _init_(self):
        self.o_width = None
        self.o_height = None
        self.o_image = None

        self.t_width = None
        self.t_height = None
        self.t_image = None
        self.save_path = None

    def remove_bg(self, file_path: str):
        self.save_path = file_path[:-4]+'.png'
        pic = Image.open(file_path)
        self.o_width = np.asarray(pic).shape[1]
        self.o_height = np.asarray(pic).shape[0]
        try:
            self.o_channels = np.asarray(pic).shape[2]
        except Exception as e:
            print("Single channel image and error", e)
        # os.remove(file_path)
        self.o_image = remove(pic)
        # self.o_image.save(self.save_path)
        # os.remove(self.save_path)
        return np.asarray(self.o_image)

    def transform(self, width=768, height=1024):
        newsize = (width, height)
        self.t_height = height
        self.t_width = width

        pic = self.o_image
        img = pic.resize(newsize)
        self.t_image = img

        background = Image.new("RGBA", newsize, (255, 255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        self.save_path = self.save_path[:-4] + 'new.jpg'
        background.convert('RGB').save(self.save_path, 'JPEG')

        return np.asarray(background.convert('RGB'))


# USAGE OF THE CLASS
# preprocess_user = preprocessInput()

# op_user = preprocess_user.remove_bg(r'C:\React\VTON\user.jpg')
# arr_user = preprocess_user.transform()

# # Image.fromarray(arr).show()

# preprocess_cloth = preprocessInput()
# op_cloth = preprocess_cloth.remove_bg(r'C:\React\VTON\cloth_img1.jpg')
# def transform(width=768, height=1024):
#         newsize = (width, height)

#         pic = op_cloth
#         img = Image.fromarray(pic).resize(newsize)
#         # print(img.split())
#         background = Image.new("RGBA", newsize, (0, 0, 0, 0))
#         background.paste(img, mask=img.split()[3])
#         save_path = r"C:\React\VTON\cloth_img1" + 'new.jpg'
#         background.convert('RGB').save(save_path, 'JPEG')

#         return np.asarray(background.convert('RGB'))
# arr_user = transform()

# Image.fromarray(op_cloth).resize((768,1024))
# Image.fromarray(op_cloth).convert("RGB").save("cloth_img1new.jpg", 'JPEG')



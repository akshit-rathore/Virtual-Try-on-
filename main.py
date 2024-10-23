from preprocess import preprocessInput

from PIL import Image
import numpy as np

preprocess_user = preprocessInput()

op_user = preprocess_user.remove_bg(r'C:\React\VTON\user.jpg')
arr_user = preprocess_user.transform()

# Image.fromarray(arr).show()

preprocess_cloth = preprocessInput()
op_cloth = preprocess_cloth.remove_bg(r'C:\React\VTON\cloth_img1.jpg')
def transform(width=768, height=1024):
        newsize = (width, height)

        pic = op_cloth
        img = Image.fromarray(pic).resize(newsize)
        # print(img.split())
        background = Image.new("RGBA", newsize, (0, 0, 0, 0))
        background.paste(img, mask=img.split()[3])
        save_path = r"C:\React\VTON\cloth_img1" + 'new.jpg'
        background.convert('RGB').save(save_path, 'JPEG')

        return np.asarray(background.convert('RGB'))
arr_user = transform()

import pose_detect

import warp_cloth
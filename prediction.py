from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from retinaface import RetinaFace

import model

members = ['上村 莉菜', '尾関 梨香', '小池 美波', '小林 由依', '齋藤 冬優花', '菅井 友香', '土生 瑞穂', '原田 葵',
       '井上 梨名', '遠藤 光莉', '大園 玲', '大沼 晶保', '幸阪 茉里乃', '関 有美子', '武元 唯衣', '田村 保乃', '藤吉 夏鈴', 
       '増本 綺良', '森田 ひかる', '松田 里奈', '守屋 麗奈', '山﨑 天']
def load_model():
    net = model.Net()
    model_path = 'model.pth'
    net.load_state_dict(torch.load(model_path))
    return net

def read_image(image_encoded: Image.Image):
    pil_image = Image.open(BytesIO(image_encoded))
    pil_image = np.asarray(pil_image)
    pil_image = cv2.cvtColor(pil_image, cv2.COLOR_RGBA2BGR)   
    return pil_image

def preprocess(images):
    resp = RetinaFace.detect_faces(images, threshold = 0.5)
    face_cut = []
    for key in resp:
        p = resp[key]
        facial_area = p["facial_area"]
        face = images[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
        face = cv2.resize(face,(50, 75))
        face= cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = np.asarray(face , np.float32)/255
        #face= cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_cut.append(face)
    face_cut = np.array(face_cut) 
    return face_cut

def predict(image: np.ndarray):
    net = load_model()
    if net is None:
        net = load_model()
    images = torch.tensor(image)
    print(images.shape)
    response = []
    with torch.no_grad():
        for im in images:
            input = im.view(1,75,50)
            outputs = net(input)
            print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            print(predicted)
            response += [members[predicted]]
    return response

import glob
import os

import cv2
import numpy as np
from PIL import Image
from retinaface import RetinaFace

import japanesefilewriteread

# メンバーの名前をリスト形式でまとめる
members = ['尾関 梨香','上村 莉菜', '小池 美波', '小林 由依', '齋藤 冬優花', '菅井 友香', '土生 瑞穂', '原田 葵',
       '井上 梨名', '遠藤 光莉', '大園 玲', '大沼 晶保', '幸阪 茉里乃', '関 有美子', '武元 唯衣', '田村 保乃', '藤吉 夏鈴', 
       '増本 綺良', '森田 ひかる', '松田 里奈', '守屋 麗奈', '山﨑 天']
            
dir1 = './data/'
dir2 = './face_cut/'
#日本語ファイルに入れるためにこのメソッドが必要
def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False
# 画像の読み込み
for member in members:
    files = glob.glob(dir1 + member + '\\' + '*.jpg')
    #print(files)
    for j, file in enumerate(files):
        print(file)
        img = Image.open(file)
        new_image = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(dir2 + member + '/' + str(j+1) + '.jpg', img)
        # 顔検出
        resp = RetinaFace.detect_faces(img, threshold = 0.5)
        if len(resp)>0:
            for key in resp:
                if key:
                    identity = resp[key]
                    print(resp[key])
                    if len(identity["facial_area"]) > 0:
                        # 検出した顔の領域範囲をトリミング
                        facial_area = identity["facial_area"]
                        #print(facial_area[1], facial_area[3], facial_area[0], facial_area[2])
                        face_cut = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
                            #cv2.imshow('sample', face_cut)
                            # 画像を保存するフォルダを作成
                        if not os.path.exists(dir2 + member):
                            os.mkdir(dir2 + member)
                        name = str(j+1)+'.jpg'
                            # 保存先のディレクトリを指定、番号を付けて保存
                        ret = japanesefilewriteread.imwrite(dir2 + member+'\\' +name, face_cut)
                        print(dir2 + member + '\\' + str(j+1) + '.jpg: ' + '検出されました')
                            # 画像を1秒間表示
                            #cv2.imshow('sample', face_cut)
                            #cv2.waitKey(1000)
                    else:
                        print(dir2 + member + '/' + str(j+1) + '.jpg' + '検出されませんでした')
                        continue
                else:
                        print(dir2 + member + '/' + str(j+1) + '.jpg' + '検出されませんでした')
                        continue
        else:
            print(dir2 + member + '/' + str(j+1) + '.jpg' + '検出されませんでした')
            continue

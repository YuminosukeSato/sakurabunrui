import glob
import os

import cv2
import numpy as np

import japanesefilewriteread

# メンバーの名前をリスト形式でまとめる
members = ['上村 莉菜', '尾関 梨香', '小池 美波', '小林 由依', '齋藤 冬優花', '菅井 友香', '土生 瑞穂', '原田 葵',
       '井上 梨名', '遠藤 光莉', '大園 玲', '大沼 晶保', '幸阪 茉里乃', '関 有美子', '武元 唯衣', '田村 保乃', '藤吉 夏鈴', 
       '増本 綺良', '森田 ひかる', '松田 里奈', '守屋 麗奈', '山﨑 天']

dir1 = './face_cut_rename/'
dir2 = './face_cut_inflated/'

def inflated_image(img, fliplrud = True,rotate=True, thr=True,resize=True, erode=True):
    # 水増しの手法を配列にまとめる
    methods = [fliplrud,rotate, thr, resize, erode]
    p = np.array(img)
    img_size = p.shape
    mat = cv2.getRotationMatrix2D(tuple(np.array([img_size[1], img_size[0]]) / 2 ), 45, 1.0)
    filter1 = np.ones((3, 3))
    # オリジナルの画像データを配列に格納
    images = [img]
    #TODO:methodの種類を増やす。
    scratch = np.array([
        #上下左右反転
        lambda x: cv2.flip(x, -1),
        #画像回転
        lambda x: cv2.warpAffine(x, cv2.getRotationMatrix2D(tuple(np.array([x.shape[1] / 2, x.shape[0] /2])), 45, 1), (x.shape[1], x.shape[0])),
        # ぼかし
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        # モザイク処理
        lambda x: cv2.resize(cv2.resize(x, (img_size[1]//6, img_size[0]//6)), (img_size[1], img_size[0])),
        #収縮
        lambda x: cv2.erode(x, filter1)    
    ])

    # 関数と画像を引数に、加工した画像を元と合わせて水増しする関数
    doubling_images = lambda f, imag: (imag + [f(i) for i in imag])
    
    # doubling_imagesを用いてmethodsがTrueの関数で水増し
    for func in scratch[methods]:
        images = doubling_images(func, images)
    
    return images

# 画像の読み込み
for i, member in enumerate(members):
    files = glob.glob(dir1 + member + '/' + '*')
    print(member)
    for j, file in enumerate(files):
        print(j)
        img = japanesefilewriteread.imread(file)
        # 画像の水増し
        inflated_images = inflated_image(img)

        # 画像を保存するディレクトリを作成
        if not os.path.exists(dir2 + member):
            os.mkdir(dir2 + member)
        # 保存先のディレクトリを指定、番号を付けて保存
        for k, im in enumerate(inflated_images):
            japanesefilewriteread.imwrite(dir2 + member + '/' + str((len(inflated_images) * j) + (k+1)) + '.jpg', im)
            #imwrite('./' + dir2 + member + '/' + str((len(inflated_images) * j) + (k+1)) + '.jpg', im)

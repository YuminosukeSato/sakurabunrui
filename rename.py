import glob
import os

#メンバーの名前をリスト形式でまとめる
members = ['上村 莉菜', '尾関 梨香', '小池 美波', '小林 由依', '齋藤 冬優花', '菅井 友香', '土生 瑞穂', '原田 葵',
       '井上 梨名', '遠藤 光莉', '大園 玲', '大沼 晶保', '幸阪 茉里乃', '関 有美子', '武元 唯衣', '田村 保乃', '藤吉 夏鈴', 
       '増本 綺良', '森田 ひかる', '松田 里奈', '守屋 麗奈', '山﨑 天']
dir2 = './face_cut/'
dir3 = './face_cut_rename/'

for member in members:
    files = glob.glob(dir2 + member + '/' + '*')
    for i, file in enumerate(files):
        # 画像を保存するフォルダーを作成
        if not os.path.exists(dir3 + member):
            os.mkdir(dir3 + member)
        # ファイル名の変更
        os.rename(file, dir3 + member + '/' + str(i+1) + '.jpg')

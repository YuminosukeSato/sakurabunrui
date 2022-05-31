import glob
import os

from icrawler.builtin import BingImageCrawler

#メンバーの名前をリスト形式でまとめる
members = ['上村 莉菜', '尾関 梨香', '小池 美波', '小林 由依', '齋藤 冬優花', '菅井 友香', '土生 瑞穂', '原田 葵',
       '井上 梨名', '遠藤 光莉', '大園 玲', '大沼 晶保', '幸阪 茉里乃', '関 有美子', '武元 唯衣', '田村 保乃', '藤吉 夏鈴', 
       '増本 綺良', '森田 ひかる', '松田 里奈', '守屋 麗奈', '山﨑 天']
dir = './data/'

for member in members:
    #指定のディレクトリを作成（画像の保存場所）
    crawler = BingImageCrawler(storage={"root_dir": dir + member})
    # 検索内容と枚数の指定
    crawler.crawl(keyword=member, max_num=500)

    #ディレクトリ内の画像ファイルを取得
    files = glob.glob(dir + member + '/' + '*')
    for i, file in enumerate(files):
        #ファイル名の変更
        os.rename(file, dir + member + '/' + str(i+1) + '.jpg')

import os
from email.mime import image
from tkinter import Y

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

import japanesefilewriteread

# メンバーの名前をリスト形式でまとめる
members = ['上村 莉菜', '尾関 梨香', '小池 美波', '小林 由依', '齋藤 冬優花', '菅井 友香', '土生 瑞穂', '原田 葵',
       '井上 梨名', '遠藤 光莉', '大園 玲', '大沼 晶保', '幸阪 茉里乃', '関 有美子', '武元 唯衣', '田村 保乃', '藤吉 夏鈴', 
       '増本 綺良', '森田 ひかる', '松田 里奈', '守屋 麗奈', '山﨑 天']
dir = './face_cut_inflated'
images = []
labels = []
number_of_members = len(members)

# 辞書にメンバーの名前とラベルを格納
members_label = {member_name: i for i, member_name in enumerate(members)}

for name in os.listdir(dir):
  # メンバーラベルの取得
  label = members_label[name]
  print(name)
  # 画像の読み込み
  for jpg in os.listdir(dir + '/' + name):
    img = japanesefilewriteread.imread(dir + '/' + name + '/' + jpg)
    # 画像のリサイズ
    #
    # cv2.imshow("sample",img)
    #cv2.waitKey(100)
    img = cv2.resize(img, (224, 224))
    # 画像データとラベルの格納
    images.append(img)
    labels.append(label)
    print(len(images))
images = np.array(images)
images = (images/255.).astype(np.float32)
#X = X.transpose(2, 0, 1).astype(np.float32)
y = np.array(labels)
  
# ホールドアウト検証  
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.20, random_state=45)
print(X_train.shape)
np.save('X_train', X_train)
np.save('X_test', X_test)
np.save('y_train', y_train)
np.save('y_test', y_test)
  # 訓練データとテストデータのラベルの可視化
print('hold-out validication:')
fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
ax.set_title('train_data')
sns.countplot(y_train, ax=ax)
ax = fig.add_subplot(1, 2, 2)
ax.set_title('test_data')
sns.countplot(y_test, ax=ax)
plt.show()

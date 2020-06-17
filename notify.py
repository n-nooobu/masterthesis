import requests
import sys
import random


'''Pythonの実行終了をLINEで通知する
https://qiita.com/aoyahashizume/items/13848b013daa18f6461b'''


def main():
    url = "https://notify-api.line.me/api/notify"
    token = 'U6suFK1EGo2900nIaAy6p7TzAqBSeBFqcUWTqs3gSqH'  # 私のLINEに通知がきてしまいます
    headers = {"Authorization": "Bearer " + token}

    index = random.randint(1, 3)
    normal_message = ['[通常終了]やったあ！',
                      '[通常終了]異常なし！イェーイ',
                      '[通常終了]スゴい！スゴい！']
    normal_image = ['image/normal end1.jpg',
                    'image/normal end2.jpg',
                    'image/normal end3.jpg']
    abnormal_message = ['[異常終了]エラー！？',
                        '[異常終了]なんでだろう...？',
                        '[異常終了]えっ？']
    abnormal_image = ['image/abnormal end1.jpg',
                      'image/abnormal end2.jpg',
                      'image/abnormal end3.jpg']

    args = sys.argv
    if args[1] == str(0):
        message = normal_message[index]
        image = normal_image[index]
    else:
        message = abnormal_message[index]
        image = abnormal_image[index]
    payload = {"message": message}
    files = {"imageFile": open(image, "rb")}

    r = requests.post(url, headers=headers, params=payload, files=files)


if __name__ == '__main__':
    args = sys.argv
    print(args)
    main()

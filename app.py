from flask import Flask
from flask import request,render_template,url_for
import cv2
from PIL import Image
import pytesseract
from matplotlib import pyplot as plt
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')
@app.route('/convert', methods=['POST'])
def convert():
    fil = request.form.get('img', None)
    tem = request.form.get('temp', None)
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract'
    img = cv2.imread(fil)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('Extract text', threshold_img)
    cv2.waitKey(0)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_NONE)
    im2 = img.copy()
    cv2.imshow('Detector', im2)
    cv2.waitKey(0)
    file = open("recognized.txt", "w+")
    file.write("")
    file.close()
    for cnt in contours:
       x, y, w, h = cv2.boundingRect(cnt)
       rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
       cropped = im2[y:y + h, x:x + w]
       file = open("recognized.txt", "a")
       text = pytesseract.image_to_string(cropped)
       file.write(text)
       file.write("\n")
       file.close
    img_rgb = cv2.imread(tem)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(fil,0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
       cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    cv2.imwrite('res.png',img_rgb)
  
    return render_template('text.html')

if __name__ == "__main__":
    app.run(debug=True)
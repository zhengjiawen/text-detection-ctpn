import os
import cv2 as cv
import numpy as np

# test img path
path = 'D:/pdfOCR/'
rotationImg1 = 'rotationImg001.png'
rotationImg2 = 'rotationImg002.png'
rotationImg3 = 'rotationImg003.png'
rotationImg4 = 'rotationImg004.png'
rotationImg5 = 'rotationImg005.png'
rotationImg6 = 'rotationImg006.png'
rotationImg7 = 'rotationImg007.png'

jpgPath = 'pdfOcrJpg/'
name = '0231.jpg'


debug = False
def drawImg(img, name):
    img = img.astype(np.uint8)
    cv.namedWindow(name, 0)
    cv.imshow(name, img)


# 收缩点团为单像素点（3×3）
#效果不好，对性能影响太大
def isolate(img):
    idx=np.argwhere(img<1)
    rows,cols=img.shape

    for i in range(idx.shape[0]):
        c_row=idx[i,0]
        c_col=idx[i,1]
        if c_col+1<cols and c_row+1<rows:
            img[c_row,c_col+1]=1
            img[c_row+1,c_col]=1
            img[c_row+1,c_col+1]=1
        if c_col+2<cols and c_row+2<rows:
            img[c_row+1,c_col+2]=1
            img[c_row+2,c_col]=1
            img[c_row,c_col+2]=1
            img[c_row+2,c_col+1]=1
            img[c_row+2,c_col+2]=1
    return img

# 输入坐标的集合
# 找到opencv官方实现了，这个deprecated
def findRectPoint(contours):

    minXIndex = np.where(contours[:,0] == np.min(contours[:,0]))
    minXList = contours[minXIndex]
    minYIndex = np.where(minXList[:,1]==np.min(minXList[:,1]))
    rectMin = minXList[minYIndex]

    maxIndex = np.where(contours[:,0] == np.max(contours[:,0]))
    maxXList = contours[maxIndex]
    maxYIndex = np.where(maxXList[:,1]==np.max(maxXList[:,1]))
    rectMax = maxXList[maxYIndex]


    return rectMin, rectMax


# 校正旋转的矩形
def rotationImg(oriImg):

    img = cv.bitwise_not(oriImg)

    # 二值化
    ret, binaryImg = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    # 开操作，消除噪声
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binaryImg = cv.morphologyEx(binaryImg, cv.MORPH_OPEN, kernel)
    # 找出像素大于0的
    coords = np.column_stack(np.where(binaryImg > 0))

    # 找出包含点集最小面积的矩形，输出是中心坐标、宽高和角度
    # angel rnge [-90,0)，所以小于45就加90进行校正，大于45直接取负数
    pos = cv.minAreaRect(coords)

    angel = pos[-1]

    if angel < -45:
        angel = -(90 + angel)
    else:
        angel = -angel


    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv.getRotationMatrix2D(center, angel, 1.0)
    rotated = cv.warpAffine(oriImg, matrix, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    if debug == True:
        print(type(matrix))
        print(matrix.shape)
        print(matrix)

    return rotated


# 校正旋转的文本，测试版
def testRotation():
    oriImg = cv.imread(path+rotationImg7, 0)
    img = cv.bitwise_not(oriImg)

    # 二值化
    ret, binaryImg = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
    #开操作，消除噪声
    kernel = cv.getStructuringElement( cv.MORPH_RECT,(3,3))
    binaryImg = cv.morphologyEx(binaryImg, cv.MORPH_OPEN, kernel)
    # 找出像素大于0的
    coords = np.column_stack(np.where(binaryImg > 0))

    if debug == True:
        print('coords size:'+ str(coords.shape))

    # 找出包含点集最小面积的矩形，输出是坐标和角度
    # angel rnge [-90,0)，所以小于45就加90进行校正，大于45直接取负数

    pos = cv.minAreaRect(coords)


    angel = pos[-1]

    if angel < -45:
        angel = -(90 + angel)
    else:
        angel = -angel

    # tempPos = (pos[0], pos[1], angel)
    box = cv.boxPoints(pos)

    if debug == True:
        testImg = oriImg.copy()
        testImg = cv.cvtColor(testImg, cv.COLOR_GRAY2RGB)
        print(pos)
        print(angel)
        print('box:'+str(box))
        print('box size:'+str(box.shape))
        a, b = pos[0]
        c, d = pos[1]
        centerX = pos[0][0]
        centerY = pos[0][1]

        width = pos[1][0]
        height = pos[1][1]

        box[:,[0,1]] = box[:,[1,0]]

        box = box.reshape((-1,1,2))
        box = box.astype(np.int)
        print('box size:'+str(box.shape))
        print(box)

        cv.polylines(testImg, [box], True, (0,0,0))
        drawImg(testImg, 'debug')
        cv.polylines(binaryImg, [box], True, 255)

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv.getRotationMatrix2D(center, angel, 1.0)
    rotated = cv.warpAffine(oriImg, matrix, (w,h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)

    if debug == True:
        print(type(matrix))
        print(matrix.shape)
        print(matrix)

    # printImg(oriImg, 'originImg')
    drawImg(binaryImg, 'binaryImg')
    drawImg(rotated, 'rotated')
    cv.waitKey()

def tableSeg(oriImg):
    '''
    分割表格
    :param oriImg: 图片，RGB
    :return: tableArray： 表格外轮廓，list, (x,y,w,h)
             rectArray:   表格cell, list,  (x,y,w,h)
    '''
    img = cv.bitwise_not(oriImg)
    thresImg = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)


    # 用腐蚀膨胀检测细长的直线
    scaleHor = 20
    scaleVec = 40
    horizontalSize = thresImg.shape[1] // scaleHor
    verticalSize = thresImg.shape[0] // scaleVec
    horKernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
    vecKernel = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))

    horImg = cv.morphologyEx(thresImg, cv.MORPH_OPEN, horKernel)
    vecImg = cv.morphologyEx(thresImg, cv.MORPH_OPEN, vecKernel)

    maskImg = horImg + vecImg

    # contours轮廓像素集，hierarchy轮廓层次信息
    contours, hierarchy = cv.findContours(maskImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 记录表格外轮廓
    tableContoursIndex = []
    tableContours = []
    # hierarchy = np.squeeze(hierarchy)

    rectContours = []
    contoursPloy = []

    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        # print(contour.shape)

        if area < 7000:
            continue
        ployContour = cv.approxPolyDP(contour, 3, True)
        contoursPloy.append(ployContour)
        boundRect = cv.boundingRect(ployContour)

        # 记录没有父轮廓，但是有子轮廓的contour，这个contour我们认为就是表格的外轮廓
        # 其他的，认为是表格每个cell的轮廓
        if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1:
            tableContoursIndex.append(i)
            tableContours.append(boundRect)
        else:
            rectContours.append(boundRect)

    rectArray = np.array(rectContours)
    tableArray = np.array(tableContours)
    if debug == True:
        print("rect array shape" + str(rectArray.shape))
        testImg = np.zeros(maskImg.shape)
        testImg = cv.bitwise_not(testImg)
        testImg = cv.cvtColor(maskImg, cv.COLOR_GRAY2RGB)
        i = 0
        for points in rectArray:
            print(points)
            i += 1
            testImg = np.zeros(maskImg.shape)
            x, y, w, h = points
            cv.rectangle(testImg, (x, y), (x + w, y + h), (255, 255, 255))

            # cv.imwrite('rectImg/' + str(i) + '.jpg', testImg)

    return tableArray, rectArray

# test for table seg
def tableSegTest():
    oriImg = cv.imread(path+name, 0)
    img = cv.bitwise_not(oriImg)
    # img = oriImg
    thresImg = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,15, -2)
    # ret, thresImg = cv.threshold(img, 0,255, cv.THRESH_OTSU)
    # thresImg = cv.bitwise_not(thresImg)

    #用腐蚀膨胀检测细长的直线
    scaleHor = 20
    scaleVec = 40
    horizontalSize = thresImg.shape[1] // scaleHor
    verticalSize = thresImg.shape[0] // scaleVec
    horKernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontalSize, 1))
    vecKernel = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalSize))

    horImg = cv.morphologyEx(thresImg, cv.MORPH_OPEN, horKernel)
    vecImg = cv.morphologyEx(thresImg, cv.MORPH_OPEN, vecKernel)

    maskImg = horImg+vecImg


    # contours轮廓像素集，hierarchy轮廓层次信息
    contours, hierarchy = cv.findContours(maskImg, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 记录表格外轮廓
    tableContoursIndex = []
    tableContours = []
    # hierarchy = np.squeeze(hierarchy)
    # print(hierarchy.shape)
    rectContours = []
    rectPointList = []
    contoursPloy = []

    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        # print(contour.shape)

        if area < 5300:
            continue
        ployContour = cv.approxPolyDP(contour, 3, True)
        contoursPloy.append(ployContour)
        boundRect = cv.boundingRect(ployContour)
        # 记录没有父轮廓，但是有子轮廓的contour，这个contour我们认为就是表格的外轮廓

        if hierarchy[0][i][3] == -1 and hierarchy[0][i][2] != -1:
            tableContoursIndex.append(i)
            tableContours.append(boundRect)
        else:
            rectContours.append(boundRect)


    rectArray = np.array(rectContours)
    # rectArray = rectArray.reshape((-1, 2,2))
    if debug == True:
        print("rect array shape"+str(rectArray.shape))
        testImg = np.zeros(maskImg.shape)
        # testImg = cv.bitwise_not(testImg)
        testImg = cv.cvtColor(maskImg, cv.COLOR_GRAY2RGB)
        i = 0
        for points in rectArray:
            print(points)
            i+=1
            # testImg = np.zeros(maskImg.shape)
            x, y, w, h= points
            cv.rectangle(testImg, (x, y), (x+w, y+h), (255,255,255))
            # cv.imwrite('rectImg/'+str(i)+'.jpg', testImg)

        cv.imwrite('rectImg/test.jpg', testImg)
        cv.imwrite('rectImg/maskImg.jpg', maskImg)
        cv.imwrite('rectImg/horImg.jpg', horImg)
        cv.imwrite('rectImg/vecImg.jpg', vecImg)


    # cv.waitKey()
# 测试将图片中的表格分割出来，原图表格部分置为白色

def splitTableTest():
    oriImg = cv.imread(path+name, 0)
    tablePointer, rectPoint = tableSeg(oriImg)
    rectPoint = rectPoint.tolist()
    rectPoint.sort(key=lambda  x:(x[1], x[0]))
    # 切割出cell
    for i, pointer in enumerate(rectPoint):
        print(pointer)
        x, y, w, h = pointer
        if h < 8 or w < 8:
            rectPoint.pop(i)
            continue
        cell = oriImg[y:y+h, x:x+w]
        cv.imwrite('cellImgTest/'+str(i)+'.jpg', cell)

    # 将表格部分置为白色
    for i, pointer in enumerate(tablePointer):
        x, y, w, h = pointer
        cv.rectangle(oriImg, (x, y), (x+w, y+h), 255, -1)
    cv.imwrite('notTableTest/'+str(i)+'.jpg', oriImg)





    # drawImg(oriImg, 'test')
    cv.waitKey()

def splitAllTableTest():
    dataPath = path+jpgPath
    # dataNum = len([name for name in os.listdir(dataPath) if os.path.isfile((os.path.join(dataPath, name)))])
    for name in os.listdir(dataPath):
        imgPath = os.path.join(dataPath, name)
        if os.path.isfile(imgPath) == False:
            continue
        oriImg = cv.imread(imgPath, 0)
        tablePointer, rectPoint = tableSeg(oriImg)
        rectPoint = rectPoint.tolist()
        rectPoint.sort(key=lambda  x:(x[1], x[0]))
        # 切割出cell
        for i, pointer in enumerate(rectPoint):
            # print(pointer)
            x, y, w, h = pointer
            if h < 8 or w < 8:
                rectPoint.pop(i)
                continue
            cell = oriImg[y:y+h, x:x+w]
            cv.imwrite('cellImg/'+name+"_"+str(i)+'.jpg', cell)

        # 将表格部分置为白色
        for i, pointer in enumerate(tablePointer):
            x, y, w, h = pointer
            cv.rectangle(oriImg, (x, y), (x+w, y+h), 255, -1)
            cv.imwrite('notTable/'+name+'.jpg', oriImg)

if __name__ == '__main__':
    splitAllTableTest()








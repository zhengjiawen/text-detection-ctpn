# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf


sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
from utils.regAndtableSeg import tableSegmentation as ts
from utils.regAndtableSeg import baiduOcr


# testDataPath = '/data/home/zjw/dataset/icdar2013/Challenge2_Test_Task12_Images/'
testDataPath = '/data/home/zjw/pythonFile/pdfOcr/pdfOcrJpg/'
# testDataPath = 'data/reImg/'
tf.app.flags.DEFINE_string('test_data_path', testDataPath, '')

# tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt/', '')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(800) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1600) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def splitTable(oriImg):
    img = oriImg.copy()
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    tablePointer, rectPoint = ts.tableSeg(grayImg)
    rectPoint = rectPoint.tolist()
    rectPoint.sort(key=lambda x: (x[1], x[0]))

    # 将表格部分置为白色
    for i, pointer in enumerate(tablePointer):
        x, y, w, h = pointer
        cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), -1)

    return img, rectPoint

def refineTable(rectPoint):
    rectPoint.sort(key=lambda  x:(x[1], x[0]))

    # 切割出cell
    resultPoint = []
    for i, pointer in enumerate(rectPoint):
        x, y, w, h = pointer
        if h < 8 or w < 8:
            continue
        resultPoint.append(pointer)

    return resultPoint


def main(argv=None):
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue
                # print(im.shape)
                
                imgWithoutTable, rectPoint = splitTable(im)
                refineRectPoint = refineTable(rectPoint)

                # imgWithoutTable = cv.cvtColor(im, cv.COLOR_GRAY2RGB)


                img, (rh, rw) = resize_image(imgWithoutTable)
                # test no resize
                # img = imgWithoutTable
                # (rh, rw) = (1,1)

                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] =boxes[:,:,0]/ rw
                    boxes[:, :, 1] = boxes[:,:,1]/rh


                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))
                #在test img 上画出bounding boxes
                ouputIm = im.copy()
                for i, box in enumerate(boxes):
                    cv2.polylines(ouputIm, [box.astype(np.int32).reshape((-1,1,2))], True, color=(0, 255, 0),
                                  thickness=2)
                for point in refineRectPoint:
                    x, y, w, h = point
                    cv2.rectangle(ouputIm, (x, y), (x + w, y + h), (255, 0, 0))
                # img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), ouputIm[:, :, ::-1])



                with open(os.path.join(FLAGS.output_path, 'res_'+os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                          "w") as f:
                    regImg = im.copy()
                    # print(regImg.shape)

                    for i, point in enumerate(refineRectPoint):
                        x, y, w, h = point
                        cellImg = regImg[y:y+h,x:x+w,:]
                        value = baiduOcr.regWordByBaiduOcr(cellImg)
                        # value = ""
                        seq = (str(x), str(y), str(x+w), str(y+h), str(1), str(value))
                        line = ",".join(seq)
                        if len(boxes) != 0 or i != len(refineRectPoint)-1:
                            line += "\r\n"
                        f.writelines(line)

                    for i, box in enumerate(boxes):
                        # print(box)
                        x, y, x2, y2 = box[0,0], box[0,1], box[2,0], box[2,1]
                        cellImg = regImg[ y:y2,x:x2, :]
                        # print(cellImg.shape)
                        # print(str(x)+" "+str(y)+" "+str(x2)+ " "+ str(y2))
                        value = baiduOcr.regWordByBaiduOcr(cellImg)
                        # value = ""
                        seq = (str(x), str(y), str(x2), str(y2), str(0), str(value))
                        line = ",".join(seq)
                        if i != len(boxes)-1:
                            line += "\r\n"

                        # line = ",".join(str(box[k]) for k in range(8))
                        # line += "," + str(scores[i]) + "\r\n"
                        f.writelines(line)



if __name__ == '__main__':
    tf.app.run()

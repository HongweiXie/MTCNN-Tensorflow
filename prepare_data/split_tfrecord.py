#coding:utf-8

import tensorflow as tf
import numpy as np
import cv2
import os

if __name__ == '__main__':

    with tf.Session() as sess:
        tfrecord_file = '/home/sixd-ailabs/Develop/Human/Caffe/data/imglists/RNet/landmark_landmark.tfrecord_shuffle'
        filename_queue = tf.train.string_input_producer([tfrecord_file], num_epochs=1, shuffle=False)
        # read tfrecord
        #batch_size=1504724
        batch_size=12
        reader = tf.TFRecordReader()
        key, serialized_example = reader.read(filename_queue)
        image_features = tf.parse_single_example(
            serialized_example,
            features={
                'image/encoded': tf.FixedLenFeature([], tf.string),  # one image  one record
                'image/label': tf.FixedLenFeature([], tf.int64),
                'image/roi': tf.FixedLenFeature([4], tf.float32),
                'image/landmark': tf.FixedLenFeature([10], tf.float32)
            }
        )
        image_size=24
        image = tf.decode_raw(image_features['image/encoded'], tf.uint8)
        image = tf.reshape(image, [image_size, image_size, 3])
        image = (tf.cast(image, tf.float32) - 127.5) / 128

        # image = tf.image.per_image_standardization(image)
        label = tf.cast(image_features['image/label'], tf.float32)
        roi = tf.cast(image_features['image/roi'], tf.float32)
        landmark = tf.cast(image_features['image/landmark'], tf.float32)

        image, label, roi, landmark = tf.train.batch(
            [image, label, roi, landmark],
            batch_size=batch_size,
            num_threads=2,
            capacity=1 * batch_size
        )
        label = tf.reshape(label, [batch_size])
        roi = tf.reshape(roi, [batch_size, 4])
        landmark = tf.reshape(landmark, [batch_size, 10])

        # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
        tf.local_variables_initializer().run()
        # 使用start_queue_runners之后，才会开始填充队列
        threads = tf.train.start_queue_runners(sess=sess)

        image_batch_array, label_batch_array, bbox_batch_array, landmark_batch_array = sess.run(
            [image, label, roi, landmark])



        i = 0
        # try:
        #     while True:
        #         i += 1
        #         # 获取图片数据并保存
        #         value = sess.run(serialized_example)
        #         print tf.parse_example(value)
        #         # print label,roi,landmark
        #         print i
        # finally:
        #     print 'xxx'

        # try:
        #     while (True):
        #         key, serialized_example = reader.read(filename_queue)
        #         # print key
        #         image_features = tf.parse_single_example(
        #             serialized_example,
        #             features={
        #                 'image/encoded': tf.FixedLenFeature([], tf.string),  # one image  one record
        #                 'image/label': tf.FixedLenFeature([], tf.int64),
        #                 'image/roi': tf.FixedLenFeature([4], tf.float32),
        #                 'image/landmark': tf.FixedLenFeature([10], tf.float32)
        #             }
        #         )
        #         roi = tf.cast(image_features['image/roi'], tf.float32)
        #         print roi
        #         cnt += 1
        #         if (cnt % 1000) == 0:
        #             print cnt
        #
        # finally:
        #     print cnt





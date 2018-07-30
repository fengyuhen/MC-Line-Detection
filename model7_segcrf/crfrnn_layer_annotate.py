"""
MIT License

Copyright (c) 2017 Sadeep Jayasumana

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import tensorflow as tf
from keras.engine.topology import Layer
import high_dim_filter_loader
custom_module = high_dim_filter_loader.custom_module


class CrfRnnLayer(Layer):
    """ Implements the CRF-RNN layer described in:

    Conditional Random Fields as Recurrent Neural Networks,
    S. Zheng, S. Jayasumana, B. Romera-Paredes, V. Vineet, Z. Su, D. Du, C. Huang and P. Torr,
    ICCV 2015
    """

    def __init__(self, image_dims, num_classes,
                 theta_alpha, theta_beta, theta_gamma,
                 num_iterations, **kwargs):
        self.image_dims = image_dims # 224
        self.num_classes = num_classes # 2
        self.theta_alpha = theta_alpha # 160
        self.theta_beta = theta_beta # 3
        self.theta_gamma = theta_gamma # 3
        self.num_iterations = num_iterations # 5
        self.spatial_kernel = None # 2x2 uniform
        self.bilateral_kernel = None # 2x2 uniform
        self.compatibility_matrix = None # 2x2 uniform
        super(CrfRnnLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Weights of the spatial kernel
        self.spatial_kernel = self.add_weight(name='spatial_kernel',
                                                   shape=(self.num_classes, self.num_classes),
                                                   initializer='uniform',
                                                   trainable=True)

        # Weights of the bilateral kernel
        self.bilateral_kernel = self.add_weight(name='bilateral_kernel',
                                                     shape=(self.num_classes, self.num_classes),
                                                     initializer='uniform',
                                                     trainable=True)

        # Compatibility matrix
        self.compatibility_matrix = self.add_weight(name='compatibility_matrix',
                                                    shape=(self.num_classes, self.num_classes),
                                                    initializer='uniform',
                                                    trainable=True)

        super(CrfRnnLayer, self).build(input_shape)

    def call(self, inputs):
        unaries = tf.transpose(inputs[0][0, :, :, :], perm=(2, 0, 1)) # 224x224x2 -> 2x224x224 分割结果
        rgb = tf.transpose(inputs[1][0, :, :, :], perm=(2, 0, 1)) # 224x224x3 -> 3x224x224 原图

        c, h, w = self.num_classes, self.image_dims[0], self.image_dims[1] # 2, 224, 224
        all_ones = np.ones((c, h, w), dtype=np.float32)

        # Prepare filter normalization coefficients
        spatial_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=False,
                                                          theta_gamma=self.theta_gamma)
        bilateral_norm_vals = custom_module.high_dim_filter(all_ones, rgb, bilateral=True,
                                                            theta_alpha=self.theta_alpha,
                                                            theta_beta=self.theta_beta)
        q_values = unaries # 2x224x224

        for i in range(self.num_iterations):
            # 5 Normalization
            softmax_out = tf.nn.softmax(q_values, 0) # 2x224x224

            # 1 Message passing
            # Spatial filtering  2X224x224 and 3x224x224 -> 2x224x224
            spatial_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=False,
                                                        theta_gamma=self.theta_gamma)
            spatial_out = spatial_out / spatial_norm_vals 

            # Bilateral filtering 2X224x224 and 3x224x224 -> 2x224x224
            bilateral_out = custom_module.high_dim_filter(softmax_out, rgb, bilateral=True,
                                                          theta_alpha=self.theta_alpha,
                                                          theta_beta=self.theta_beta)
            bilateral_out = bilateral_out / bilateral_norm_vals 

            # 2 Weighting filter outputs # 2x2 * 2x50176 + 2x2 * 2x50176
            reweight = (tf.matmul(self.spatial_kernel, tf.reshape(spatial_out, (c, -1))) +   
                        tf.matmul(self.bilateral_kernel, tf.reshape(bilateral_out, (c, -1)))) 

            # 3 Compatibility transform   2x2 * 2x50176 = 2x50176
            pairwise = tf.matmul(self.compatibility_matrix, reweight)

            # 4 Adding unary potentials
            pairwise = tf.reshape(pairwise, (c, h, w)) # 2x50176 -> 2x224x224
            q_values = unaries - pairwise # 2x224x224

        # 1x2x224x224 -> 1x224x224x2    
        return tf.transpose(tf.reshape(q_values, (1, c, h, w)), perm=(0, 2, 3, 1))

    def compute_output_shape(self, input_shape):
        return input_shape

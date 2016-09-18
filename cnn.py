import numpy as np
import sys

def conv2(X, k):
    # as a demo code, here we ignore the shape check
    x_row, x_col = X.shape
    k_row, k_col = k.shape
    ret_row, ret_col = x_row - k_row + 1, x_col - k_col + 1
    ret = np.empty((ret_row, ret_col))
    for y in range(ret_row):
        for x in range(ret_col):
            sub = X[y : y + k_row, x : x + k_col]
            ret[y,x] = np.sum(sub * k)
    return ret

def rot180(in_data):
    ret = in_data.copy()
    yEnd = ret.shape[0] - 1
    xEnd = ret.shape[1] - 1
    for y in range(ret.shape[0] / 2):
        for x in range(ret.shape[1]):
            ret[yEnd - y][x] = ret[y][x]
    for y in range(ret.shape[0]):
        for x in range(ret.shape[1] / 2):
            ret[y][xEnd - x] = ret[y][x]
    return ret

def padding(in_data, size):
    cur_r, cur_w = in_data.shape[0], in_data.shape[1]
    new_r = cur_r + size * 2
    new_w = cur_w + size * 2
    ret = np.zeros((new_r, new_w))
    ret[size:cur_r + size, size:cur_w+size] = in_data
    return ret

def discreterize(in_data, size):
    num = in_data.shape[0]
    ret = np.zeros((num, size))
    for i, idx in enumerate(in_data):
        ret[i, idx] = 1
    return ret

class ConvLayer:
    def __init__(self, in_channel, out_channel, kernel_size, lr=0.01, momentum=0.9, name='Conv'):
        self.w = np.random.randn(in_channel, out_channel, kernel_size, kernel_size)
        self.b = np.zeros((out_channel))
        self.layer_name = name
        self.lr = lr
        self.momentum = momentum

        self.prev_gradient_w = np.zeros_like(self.w)
        self.prev_gradient_b = np.zeros_like(self.b)
    # def _relu(self, x):
    #     x[x < 0] = 0
    #     return x
    def forward(self, in_data):
        # assume the first index is channel index
        print 'conv forward:' + str(in_data.shape)
        in_batch, in_channel, in_row, in_col = in_data.shape
        out_channel, kernel_size = self.w.shape[1], self.w.shape[2]
        self.top_val = np.zeros((in_batch, out_channel, in_row - kernel_size + 1, in_col - kernel_size + 1))
        self.bottom_val = in_data

        for b_id in range(in_batch):
            for o in range(out_channel):
                for i in range(in_channel):
                    self.top_val[b_id, o] += conv2(in_data[b_id, i], self.w[i, o])
                self.top_val[b_id, o] += self.b[o]
        return self.top_val

    def backward(self, residual):
        in_channel, out_channel, kernel_size = self.w.shape
        in_batch = residual.shape[0]
        # gradient_b        
        self.gradient_b = residual.sum(axis=3).sum(axis=2).sum(axis=0) / self.batch_size
        # gradient_w
        self.gradient_w = np.zeros_like(self.w)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    self.gradient_w[i, o] += conv2(self.bottom_val[b_id], residual[o])
        self.gradient_w /= self.batch_size
        # gradient_x
        gradient_x = np.zeros_like(self.bottom_val)
        for b_id in range(in_batch):
            for i in range(in_channel):
                for o in range(out_channel):
                    gradient_x[b_id, i] += conv2(padding(residual, kernel_size - 1), rot180(self.w[i, o]))
        gradient_x /= self.batch_size
        # update
        self.prev_gradient_w = self.prev_gradient_w * self.momentum - self.gradient_w
        self.w += self.lr * self.prev_gradient_w
        self.prev_gradient_b = self.prev_gradient_b * self.momentum - self.gradient_b
        self.b += self.lr * self.prev_gradient_b
        return gradient_x

class FCLayer:
    def __init__(self, in_num, out_num, lr = 0.01, momentum=0.9):
        self._in_num = in_num
        self._out_num = out_num
        self.w = np.random.randn(in_num, out_num)
        self.b = np.zeros((out_num, 1))
        self.lr = lr
        self.momentum = momentum
        self.prev_grad_w = np.zeros_like(self.w)
        self.prev_grad_b = np.zeros_like(self.b)
    # def _sigmoid(self, in_data):
    #     return 1 / (1 + np.exp(-in_data))
    def forward(self, in_data):
        print 'fc forward=' + str(in_data.shape)
        self.topVal = np.dot(self.w.T, in_data) + self.b
        self.bottomVal = in_data
        return self.topVal
    def backward(self, loss):
        batch_size = loss.shape[0]

        # residual_z = loss * self.topVal * (1 - self.topVal)
        grad_w = np.dot(self.bottomVal, loss.T) / batch_size
        grad_b = np.sum(loss) / batch_size
        residual_x = np.dot(self.w, loss)
        self.prev_grad_w = self.prev_grad_w * momentum - grad_w
        self.prev_grad_b = self.prev_grad_b * momentum - grad_b
        self.w -= self.lr * self.prev_grad_w
        self.b -= self.lr * self.prev_grad_b
        return residual_x

class ReLULayer:
    def __init__(self, name='ReLU'):
        pass

    def forward(self, in_data):
        self.top_val = in_data
        ret = in_data.copy()
        ret[ret < 0] = 0
        return ret
    def backward(self, residual):
        gradient_x = residual.copy()
        gradient_x[self.top_val < 0] = 0
        return gradient_x

class MaxPoolingLayer:
    def __init__(self, kernel_size, name='MaxPool'):
        self.kernel_size = kernel_size

    def forward(self, in_data):
        in_batch, in_channel, in_row, in_col = in_data.shape
        k = self.kernel_size
        out_row = in_row / k + (1 if in_row % k != 0 else 0)
        out_col = in_col / k + (1 if in_col % k != 0 else 0)

        self.flag = np.zeros_like(in_data)
        ret = np.empty((in_batch, in_channel, out_row, out_col))
        for b_id in range(in_batch):
            for c in range(in_channel):
                for oy in range(out_row):
                    for ox in range(out_col):
                        height = k if (oy + 1) * k <= in_row else in_row - oy * k
                        width = k if (ox + 1) * k <= in_col else in_col - ox * k
                        idx = np.argmax(in_data[b_id, c, oy * k: oy * k + height, ox * k: ox * k + width])
                        offset_r = idx / width
                        offset_c = idx % width
                        self.flag[b_id, c, oy * k + offset_r, ox * k + offset_c] = 1                        
                        ret[b_id, c, oy, ox] = in_data[b_id, c, oy * k + offset_r, ox * k + offset_c]
        return ret
    def backward(self, residual):
        in_batch, in_channel, in_row, in_col = self.flag
        k = self.kernel_size
        out_row, out_col = residual.shape[2], residual.shape[3]

        gradient_x = np.zeros_like(self.flag)
        for b_id in range(in_batch):
            for c in range(in_channel):
                for oy in range(out_row):
                    for ox in range(out_col):
                        height = k if (oy + 1) * k <= in_row else in_row - oy * k
                        width = k if (ox + 1) * k <= in_col else in_col - ox * k
                        gradient_x[b_id, c, oy * k + offset_r, ox * k + offset_c] = residual[b_id, c, oy, ox]
        gradient_x[self.flag == 0] = 0
        return gradient_x

class FlattenLayer:
    def __init__(self, name='Flatten'):
        pass
    def forward(self, in_data):
        self.in_batch, self.in_channel, self.r, self.c = in_data.shape
        return in_data.reshape(self.in_batch, self.in_channel * self.r * self.c)
    def backward(self, residual):
        return residual.reshape(self.in_batch, self.in_channel, self.r, self.c)

class SoftmaxLayer:
    def __init__(self, name='Softmax'):
        pass
    def forward(self, in_data):
        exp_out = np.exp(in_data)
        self.top_val = exp_out / np.sum(exp_out, axis=1)
        return self.top_val
    def backward(self, residual):
        return self.top_val - residual

class Net:
    def __init__(self):
        self.layers = []
    def addLayer(self, layer):
        self.layers.append(layer)
    def train(self, trainData, trainLabel, validData, validLabel, batch_size, iteration):
        train_num = trainData.shape[0]
        for iter in range(iteration):
            print 'iter=' + str(iter)
            for batch_iter in range(0, train_num, batch_size):
                if batch_iter + batch_size < train_num:
                    self.train_inner(trainData[batch_iter: batch_iter + batch_size],
                        trainLabel[batch_iter: batch_iter + batch_size])
                else:
                    self.train_inner(trainData[batch_iter: train_num],
                        trainLabel[batch_iter: train_num])
            print "eval=" + str(self.eval(validData, validLabel))
    def train_inner(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        residual_in = label
        for i in range(0, lay_num, -1):
            residual_out = self.layers[i].backward(residual_in)
            residual_in = residual_out
    def eval(self, data, label):
        lay_num = len(self.layers)
        in_data = data
        for i in range(lay_num):
            out_data = self.layers[i].forward(in_data)
            in_data = out_data
        out_idx = np.argmax(in_data, axis=1)
        label_idx = np.argmax(label, axis=1)
        return np.sum(out_idx == label_idx) / float(out_idx.shape[0])

if __name__ == '__main__':
    import struct
    from array import array

    def load_data(data_path, label_path):
        with open(label_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                    'got %d' % magic)
            labels = array("B", file.read())
        with open(data_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                    'got %d' % magic)
            image_data = array("B", file.read())
        images = []
        for i in xrange(size):
            images.append([0]*rows*cols)
        for i in xrange(size):
            images[i][:] = image_data[i*rows*cols : (i+1)*rows*cols]
        return np.array(images), np.array(labels)

    train_feature_raw, train_label_raw = load_data('train.feat', 'train.label')
    valid_feature_raw, valid_label_raw = load_data('valid.feat', 'valid.label')
    print 'load ok'
    train_feature = train_feature_raw.reshape(60000, 1, 28, 28)
    valid_feature = valid_feature_raw.reshape(10000, 1, 28, 28)
    train_label = discreterize(train_label_raw, 10)
    valid_label = discreterize(valid_label_raw, 10)

    net = Net()
    net.addLayer(ConvLayer(1, 20, 4, 0.01, 0.9))
    net.addLayer(ReLULayer())
    net.addLayer(MaxPoolingLayer(2))
    
    net.addLayer(ConvLayer(20, 40, 5, 0.01, 0.9))
    net.addLayer(ReLULayer())
    net.addLayer(MaxPoolingLayer(3))
    
    net.addLayer(FlattenLayer())
    net.addLayer(FCLayer(40 * 3 * 3, 150, 0.01, 0.9))
    net.addLayer(ReLULayer())
    net.addLayer(FCLayer(150, 10, 0.01, 0.9))
    net.addLayer(SoftmaxLayer())
    print 'net build ok'
    net.train(train_feature, train_label, valid_feature, valid_label, 100 ,10)
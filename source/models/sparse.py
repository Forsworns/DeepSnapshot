import torch.nn as nn

# from xc
class SparseNet(nn.Module):
    def __init__(self):
        super(SparseNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


def build_one_phase(layerxk, layerzk, Phi, PhiT, Yinput):
    # params
    lambdaStep = tf.Variable(0.1, dtype=tf.float32)
    softThr = tf.Variable(0.1, dtype=tf.float32)
    t = tf.Variable(1, dtype=tf.float32)
    convSize1 = 64
    convSize2 = 64
    convSize3 = 64
    filterSize1 = 3
    filterSize2 = 3
    filterSize3 = 3 

    # get rk from zk
    rk = tf.reduce_sum(tf.multiply(Phi, layerzk[-1]), axis=3)
    rk = tf.reshape(rk, shape=[-1, pixel, pixel, 1])
    rk = tf.subtract(rk, Yinput)
    rk = tf.multiply(PhiT, tf.tile(rk, [1, 1, 1, nFrame]))
    rk = tf.scalar_mul(lambdaStep, rk)
    rk = tf.subtract(layerzk[-1], rk)

    # F(rk)
    weight0 = get_filter([filterSize1, filterSize1, nFrame, convSize1], 0)
    weight11 = get_filter([filterSize2, filterSize2, convSize1, convSize2], 11)
    weight12 = get_filter([filterSize3, filterSize3, convSize2, convSize3], 12)
    Frk = tf.nn.conv2d(rk, weight0, strides=[1, 1, 1, 1], padding='SAME')
    tmp = Frk
    Frk = tf.nn.conv2d(Frk, weight11, strides=[1, 1, 1, 1], padding='SAME')
    Frk = tf.nn.relu(Frk)
    Frk = tf.nn.conv2d(Frk, weight12, strides=[1, 1, 1, 1], padding='SAME')

    # soft threshold, soft(F(rk), softThr)
    softFrk = tf.multiply(tf.sign(Frk), tf.nn.relu(tf.subtract(tf.abs(Frk), softThr)))    

    # ~F(soft(F(rk), softThr))
    weight13 = get_filter([filterSize3, filterSize3, convSize3, convSize2], 53)
    weight14 = get_filter([filterSize2, filterSize2, convSize2, convSize1], 54)
    weight6 = get_filter([filterSize1, filterSize1, convSize1, nFrame], 6)
    FFrk = tf.nn.conv2d(softFrk, weight13, strides=[1, 1, 1, 1], padding='SAME')
    FFrk = tf.nn.relu(FFrk)
    FFrk = tf.nn.conv2d(FFrk, weight14, strides=[1, 1, 1, 1], padding='SAME')
    FFrk = tf.nn.conv2d(FFrk, weight6, strides=[1, 1, 1, 1], padding='SAME')

    # xk = rk + ~F(soft(F(rk), softThr))
    xk = tf.add(rk, FFrk)
    zk = (1 + t)*xk - t*layerxk[-1]

    # Symmetric constraint
    sFFrk = tf.nn.conv2d(Frk, weight13, strides=[1, 1, 1, 1], padding='SAME')
    sFFrk = tf.nn.relu(sFFrk)
    sFFrk = tf.nn.conv2d(sFFrk, weight14, strides=[1, 1, 1, 1], padding='SAME')
    symmetric = sFFrk - tmp
    return xk, zk, symmetric, Frk

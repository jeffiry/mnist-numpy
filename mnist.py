import numpy as np
import dnn
import os
import matplotlib.pyplot as plt
import matplotlib


def save(parameters, save_as):
    dic = {}
    for i in range(len(parameters)):
        dic[str(i)] = parameters[i].data
    np.savez(save_as, **dic)


def load(parameters, file):
    params = np.load(file)
    for i in range(len(parameters)):
        parameters[i].data = params[str(i)]


def load_MNIST(file, transform=False):
    file = np.load(file)
    X = file['X']
    Y = file['Y']
    if transform:
        X = X.reshape(len(X), -1)
    return X, Y


def train(net, loss_fn, train_file, test_file, batch_size, optimizer, load_file, save_as, times=1, retrain=False, l2norm=0.0):
    X, Y = load_MNIST(train_file, transform=True)
    data_size = X.shape[0]
    if not retrain and os.path.isfile(load_file):
        print('%s is exited' % load_file)
        load(net.parameters, load_file)
        test_acc, test_loss = test(net, loss_fn, test_file)
        print('Best performace on test dataset, acc:%.2f loss:%2.2f' % (test_acc * 100, test_loss))
        return
    best_acc, best_model = 0.0, None
    patience, cur = 20, 0
    earlystopping = False
    accs, trainlosses, testlosses = [], [], []
    for loop in range(times):
        i = 0
        while i <= data_size - batch_size:
            x = X[i:i + batch_size]
            y = Y[i:i + batch_size]
            i += batch_size

            output = net.forward(x)
            batch_acc, batch_loss = loss_fn(output, y)
            origin_loss = batch_loss
            l2loss = net.getL2Norm()
            batch_loss += l2loss * l2norm

            eta = loss_fn.gradient()
            net.backward(eta)
            optimizer.update()
            if i % 10 == 0:
                valid_acc, valid_loss = test(net, loss_fn, test_file)
                print("loop: %d, batch: %5d, batch acc: %2.4f, batch loss: %.2f, l2loss:%.2f, valid acc: %2.4f, valid loss: %.2f" % \
                      (loop, i, batch_acc * 100, batch_loss, l2loss * l2norm, valid_acc * 100, valid_loss))
                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = net.parameters
                    cur = 0
                else:
                    cur += 1
                    if cur >= patience:
                        earlystopping = True
                        break
                accs.append(valid_acc)
                trainlosses.append(origin_loss)
                testlosses.append(valid_loss)
        if earlystopping:
            break
    if save_as is not None:
        save(best_model, save_as)
        load(net.parameters, load_file)
        test_acc, test_loss = test(net, loss_fn, test_file)
        print('Best performace on test dataset, acc:%.2f loss:%2.2f' % (test_acc * 100, test_loss))
    visualization(accs, trainlosses, testlosses, data_size, batch_size)


def test(net, loss_fn, test_file):
    X, Y = load_MNIST(test_file, transform=True)
    output = net.forward(X)
    acc, loss = loss_fn(output, Y)
    return acc, loss


def visualization(accs, trainlosses, testlosses, dsize, bsize):
    # print(dsize, bsize, len(accs))
    epochs = [i/(dsize//(bsize*10)) for i in range(1, len(accs)+1)]
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, testlosses, color='r', label='test loss')
    plt.plot(epochs, trainlosses, color='royalblue', label='train loss')
    plt.legend()
    plt.xlabel("迭代次数/次")
    plt.ylabel("误差(loss)")
    plt.title("loss-迭代次数")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accs, color='orange', label='test accuracy')
    plt.legend()
    plt.xlabel("迭代次数/次")
    plt.ylabel("预测准确度")
    plt.title("accuracy-迭代次数")
    plt.show()


def search():
    print('Please input lr, h, l2loss\nFor example: 0.0001 400 100 0.0001')
    print('lr can be chosen from 5e-5, 2e-4, 5e-4, 1e-3')
    print('h can be chosen from 400 200 100')
    print('l2 parameter can be chosen from 2e-5 5e-5')
    print('If model searched exists, the performance will be shown. If not, the model will be trained and showned.')
    lr, h, l2loss = map(float, input().split())
    h = int(h)
    return lr, h, l2loss

if __name__ == "__main__":
    lr, h, l2loss = search()
    h = 400
    layers = [
        {'type': 'batchnorm', 'shape': 784, 'requires_grad': False, 'affine': False},
        {'type': 'linear', 'shape': (784, h)},
        {'type': 'batchnorm', 'shape': h},
        {'type': 'relu'},
        # {'type': 'linear', 'shape': (h1, h2)},
        # {'type': 'batchnorm', 'shape': h2},
        # {'type': 'relu'},
        {'type': 'linear', 'shape': (h, 10)}
    ]
    loss_fn = dnn.CrossEntropyLoss()
    net = dnn.Net(layers)
    lr = 5e-3
    batch_size = 256
    l2 = 1e-5
    optimizer = dnn.Adam(net.parameters, lr)
    train_file = './dataset/trainset.npz'
    test_file = './dataset/testset.npz'
    savefile = './models/lr_%.4f - h_%d - l2_%.5f' % (lr, h, l2)
    loadfile = './models/lr_%.4f - h_%d - l2_%.5f.npz' % (lr, h, l2)
    train(net, loss_fn, train_file, test_file, batch_size, optimizer, loadfile, savefile, times=20, retrain=False, l2norm=l2)

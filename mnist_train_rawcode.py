import numpy as np
import gzip
import struct
import matplotlib.pyplot as plt

#读取数据————————————————————————————————
def load_images(filename):
    with gzip.open(filename,'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII',f.read(16)) # type: ignore
        data = np.frombuffer(f.read(),dtype=np.uint8)
        images = data.reshape(num,rows, cols)
    return images

def load_labels(filename):
    with gzip.open(filename,'rb')as f:
        magic, num = struct.unpack(">II",f.read(8))
        labels = np.frombuffer(f.read(),dtype=np.uint8)
        return labels

train_images = load_images('./data/MNIST/raw/train-images-idx3-ubyte.gz')
train_labels = load_labels('./data/MNIST/raw/train-labels-idx1-ubyte.gz')

#数据预处理——————————————————————————————
train_images = train_images.astype(np.float32)/255.0

train_images = train_images.reshape(train_images.shape[0],-1)

def one_hot(labels, num_classes = 10):
    result = np.zeros((labels.size, num_classes))
    result[np.arange(labels.size),labels] = 1
    return result

train_labels_oh = one_hot(train_labels)

#神经网络————————————————————————————————
input_size = 784
hidden_size = 128
output_size = 10

np.random.seed(0)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    exp_x = np.exp(x - np.max(x,axis=1,keepdims=True))
    return exp_x / np.sum(exp_x,axis=1,keepdims=True)

def forward(x):
    z1 = np.dot(x, W1) +b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) +b2
    a2 = softmax(z2)
    return z1, a1 ,z2, a2

#反向传播+更新————————————————————————
def cross_entropy_loss(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred,eps,1. -eps)
    N = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred)) / N
    return loss

def backward(x,y_true,z1,a1,z2,a2):
    global W1,b1,W2,b2

    m = x.shape[0]

    dz2 = (a2 - y_true) / m
    dW2 = np.dot(a1.T,dz2)
    db2 = np.sum(dz2,axis=0,keepdims=True)

    da1 = np.dot(dz2,W2.T)
    dz1 = da1 * (z1>0)
    dW1 = np.dot(x.T, dz1)
    db1 = np.sum(dz1, axis=0,keepdims=True)

    lr = 0.01
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

epochs = 60
batch_size = 256
loss_history = []

for epoch in range(epochs):
    permutation = np.random.permutation(train_images.shape[0])
    train_images_shuffled = train_images[permutation]
    train_labels_shuffled = train_labels_oh[permutation]
    
    epoch_loss = 0
    batches = 0


    for i in range(0, train_images.shape[0], batch_size):
        x_batch = train_images_shuffled[i:i+batch_size]
        y_batch = train_labels_shuffled[i:i+batch_size]

        z1, a1, z2, a2 = forward(x_batch)
        loss = cross_entropy_loss(y_batch, a2)
        backward(x_batch, y_batch,z1, a1, z2, a2)

        epoch_loss += loss
        batches += 1

    # loss_history.append(loss)
    avgloss = epoch_loss / batches
    loss_history.append(avgloss)
    print(f"Epoch{epoch+1},loss = {loss:.4f}")

#准确率确认—————————————————————————————————
def accuracy(x,y_true):
    _,_,_,a2 = forward(x)
    y_pred = np.argmax(a2,axis=1)
    y_true_labels = np.argmax(y_true, axis =1)
    return np.mean(y_pred == y_true_labels)

print("训练集准确率:", accuracy(train_images[:10000], train_labels_oh[:10000]))

plt.plot(range(1,epochs+1), loss_history,marker='o')
plt.title('Loss Curve (Training)')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()

#测试+画图——————————————————————————————————
test_images = load_images('./data/MNIST/raw/t10k-images-idx3-ubyte.gz')
test_labels = load_labels('./data/MNIST/raw/t10k-labels-idx1-ubyte.gz')

test_images = test_images.astype(np.float32) / 255.0
test_images = test_images.reshape(test_images.shape[0], -1)
test_labels_oh = one_hot(test_labels)

test_acc = accuracy(test_images, test_labels_oh)
print(f"测试集准确率: {test_acc * 100:.2f}%")
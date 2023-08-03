import numpy as np
import matplotlib.pyplot as plt
#Lấy và phần tích dữ liệu bằng thư viện Keras
from keras.datasets import cifar10 #cifar10 là bộ ảnh

#Tải ảnh
#X điểm màu
#Y nhãn
#Xtran là tâm hình, YTrain là nhãn
(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

print(Xtrain.shape) #50000 ngàn tấm hình kích thước 32 32 Với 3 layer (RGB)

# Y Train sẻ ra một số đại diện cho nhãn
# mình định nghĩa số đó là nhãn gì
labels = ["Nhan 0", "Nhan 1", "Nhan 2", "Nhan 3", "Nhan 4", "Nhan 5", "Nhan 6", "Nhan 7", "Nhan 8", "Nhan 9", "Nhan 10"]

def show_image(data, number):
    for i in range(number):
        plt.subplot(5, 10, i+1)
        plt.imshow(data[0 + i]) #Nếu lấu từ vị trí thứ 2 và lấy 50 tấm thì thay số 0 bằng số 2
        idx_label = Ytrain[0 + i][0]
        plt.title(labels[idx_label])
        plt.axis("off")
    plt.show()

show_image(Xtrain, 50)
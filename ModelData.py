import numpy as np
import matplotlib.pyplot as plt
#Lấy và phần tích dữ liệu bằng thư viện Keras
from keras.datasets import cifar10 #cifar10 là bộ ảnh
from keras import layers
from keras import models

#Tải ảnh
#X điểm màu
#Y nhãn
#Xtran là tâm hình, YTrain là nhãn
(Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

"""
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
"""

#Sequential Tạo chuổi
#layers.Flatten(input_shape=(32, 32)) 32, 32 đây là kích thước tấm hình
model_training_demo = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(32 * 32, activation='relu'), #32 * 32 Khéo về dạng 32*32 nhãn
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax'),
    #Mục đích của 3 cái giảm dần là sử dụng Fylly connective để đưa về số lượng nhãn bé nhất
])

model_training_demo.summary()

"""
 flatten (Flatten)           (None, 3072) 32 * 32 * 3             0         
                                                                 
 dense (Dense)               (None, 1024)  32 * 32            3146752   Nối fullu connective của 3072 điểm với 1024 điểm
                                                                 
 dense_1 (Dense)             (None, 1000)              1025000   1024 * 1000 + 1000 bias
                                                                 
 dense_2 (Dense)             (None, 10)                10010  
 
 Total params: 4181762 (15.95 MB) Bằng tổng của tất cả cột cuối cùng
Trainable params: 4181762 (15.95 MB)
Non-trainable params: 0 (0.00 Byte)
"""
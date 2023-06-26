# 导入需要用到的库
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def train_xx():
    # 执行清空环境变量的操作
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.keras.backend.clear_session()

    # 读取数据并进行分析
    df = pd.read_excel('shuju2.xlsx', header=None)
    res = df.to_numpy()
    num_class = len(np.unique(res[:, -1]))
    num_dim = res.shape[1] - 1
    num_res = res.shape[0]
    num_size = 0.8
    flag_confusion = True

    # 分离出训练集和测试集，并进行归一化处理
    np.random.shuffle(res)
    train_data, test_data = np.split(res, [int(num_res * num_size)])
    scaler = MinMaxScaler(feature_range=(0, 1))
    P_train = scaler.fit_transform(train_data[:, :-1])
    T_train = to_categorical(train_data[:, -1] - 1, num_classes=num_class)
    P_test = scaler.transform(test_data[:, :-1])
    T_test = to_categorical(test_data[:, -1] - 1, num_classes=num_class)

    # 将数据转置并平铺为适合输入网络的形状
    P_train = P_train.reshape(P_train.shape[0], P_train.shape[1], 1, 1)
    T_train = T_train
    P_test = P_test.reshape(P_test.shape[0], P_test.shape[1], 1, 1)
    T_test = T_test

    # 构造卷积神经网络结构
    x_input = Input(shape=(num_dim, 1, 1))
    x = Conv2D(filters=16, kernel_size=(2, 1), padding='valid', activation=None)(x_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=1)(x)
    x = Conv2D(filters=32, kernel_size=(2, 1), padding='valid', activation=None)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 1), strides=1)(x)
    x = Flatten()(x)
    output = Dense(num_class, activation='softmax')(x)
    model = Model(inputs=x_input, outputs=output)

    # 设置并训练模型
    opt = tf.keras.optimizers.Adam(learning_rate=2e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    print(np.shape(P_train))
    print(np.shape(T_train))
    print(np.shape(P_test))
    print(np.shape(T_test))
    model.fit(P_train, T_train, epochs=600, batch_size=32, validation_split=0.2, shuffle=True, verbose=0)


# model.save('model.h5')

def predict(data):
    model = load_model('./model.h5')
    result = np.argmax(model.predict(data), axis=1) + 1
    return result


if __name__ == '__main__':
    data = np.array([0.34, 0.324, 0.4532, 0.678, 0.768, 0.1654, 0.8764, 0.7459, 1]).reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    pre_data = scaler.fit_transform(data)
    print(pre_data.shape)
    print(pre_data)
    pre_data = pre_data.reshape(pre_data.shape[1], pre_data.shape[0], 1, 1)
    # print(pre_data)
    result = predict(pre_data)
    print(result)

# # 预测模型，并计算错误率
# T_sim1 = np.argmax(model.predict(P_train), axis=1) + 1
# T_sim2 = np.argmax(model.predict(P_test), axis=1) + 1
# error1 = sum((T_sim1 == train_data[:, -1])) / len(train_data) * 100
# error2 = sum((T_sim2 == test_data[:, -1])) / len(test_data) * 100
#
# # 绘图并输出混淆矩阵
# plt.plot(history.history['accuracy'], label='train accuracy')
# plt.plot(history.history['val_accuracy'], label='val accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Train and Validation Accuracy')
# plt.legend()
# plt.show()
#
# plt.plot(history.history['loss'], label='train loss')
# plt.plot(history.history['val_loss'], label='val loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Train and Validation Loss')
# plt.legend()
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(T_sim1)), train_data[:, -1], 'r-*', label='Real Value')
# ax.plot(np.arange(len(T_sim1)), T_sim1, 'b-o', label='Predicted Value')
# ax.set_xlabel('Sample')
# ax.set_ylabel('Result')
# ax.set_title('Train Data Prediction Comparison\nAccuracy: {:.2f}%'.format(error1))
# ax.legend()
# ax.grid(True)
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(T_sim2)), test_data[:, -1], 'r-*', label='Real Value')
# ax.plot(np.arange(len(T_sim2)), T_sim2, 'b-o', label='Predicted Value')
# ax.set_xlabel('Sample')
# ax.set_ylabel('Result')
# ax.set_title('Test Data Prediction Comparison\nAccuracy: {:.2f}%'.format(error2))
# ax.legend()
# ax.grid(True)
# plt.show()
#
# if flag_confusion:
#     cm_train = confusion_matrix(train_data[:, -1], T_sim1)
#     print('Confusion Matrix for Train Data:\n', cm_train)
#     cm_test = confusion_matrix(test_data[:, -1], T_sim2)
#     print('Confusion Matrix for Test Data:\n', cm_test)

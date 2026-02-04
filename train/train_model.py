import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def train():
    print("1. 正在加载 CIFAR-10 数据集...")
    print("提示：如果下载过慢，请手动下载 'cifar-10-python.tar.gz'")
    print("放入 C:\\Users\\你的用户名\\.keras\\datasets\\ 并重命名为 'cifar-10-batches-py.tar.gz'")
    
    # 加载 CIFAR-10 数据
    # Keras 会先检查本地 .keras/datasets/ 目录下是否有文件
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 数据预处理：归一化到 0-1 之间
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    print(f"训练集数量: {len(x_train)}, 测试集数量: {len(x_test)}")

    # 2. 构建卷积神经网络 (CNN)
    print("2. 构建模型结构 (含数据增强)...")
    
    # 定义数据增强层 (对应 PyTorch 的 transforms)
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"), # 随机水平翻转
        layers.RandomRotation(0.1),      # 随机旋转 +/- 10%
        layers.RandomZoom(0.1),          # 随机缩放
    ])

    model = models.Sequential([
        # 输入层
        layers.Input(shape=(32, 32, 3)),
        
        # 加入数据增强层 (只在训练时生效，预测时自动跳过)
        data_augmentation,
        
        # 第一层卷积：32个 3x3 卷积核，ReLU 激活
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 第二层卷积：64个 3x3 卷积核
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # 第三层卷积
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        
        # 展平层
        layers.Flatten(),
        
        # 全连接层
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(10, activation='softmax')
    ])

    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 3. 训练模型
    print("3. 开始训练 (预计需要几分钟)...")
    history = model.fit(x_train, y_train, 
                        epochs=10, # 训练 10 轮
                        validation_data=(x_test, y_test),
                        batch_size=64)

    # 4. 保存模型
    save_dir = os.path.join(os.path.dirname(__file__), '../cloudfunctions/animalPredict/model')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, 'cifar_model.h5')
    model.save(save_path)
    print(f"4. 模型已保存至: {save_path}")
    print("请回到微信开发者工具，右键 'animalPredict' 文件夹 -> 上传并部署")

if __name__ == '__main__':
    # 检查 TensorFlow 版本
    print(f"TensorFlow Version: {tf.__version__}")
    train()
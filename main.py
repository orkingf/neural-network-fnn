import cv2
from multiprocessing import Pool
import numpy as np
import os
def split_data_k_fold(data_path, k=5):
    # 获取数据路径下的所有文件夹（每个文件夹代表一个字符或类别）
    character_folders = os.listdir(data_path)

    # 初始化文件列表，每个 fold 一个训练和测试文件
    files_train = [open(f'train_data_fold_{i+1}.txt', 'w') for i in range(k)]
    files_test = [open(f'test_data_fold_{i+1}.txt', 'w') for i in range(k)]

    # 遍历每个字符文件夹
    for character_folder in character_folders:
        folder_path = os.path.join(data_path, character_folder)
        if os.path.isdir(folder_path):  # 确保是目录
            images = [img for img in os.listdir(folder_path) if img.endswith('.JPG')]
            np.random.shuffle(images)  # 随机打乱

            # 分割数据集为 k 份
            num_images = len(images)
            fold_size = num_images // k
            remainder = num_images % k
            start = 0

            # 分配数据到 k 折
            for i in range(k):
                end = start + fold_size + (1 if i < remainder else 0)
                test_images = images[start:end]
                train_images = images[:start] + images[end:]

                # 写入对应的文件
                for img in train_images:
                    line = os.path.join(folder_path, img) + '\t' + character_folder + '\n'
                    files_train[i].write(line)

                for img in test_images:
                    line = os.path.join(folder_path, img) + '\t' + character_folder + '\n'
                    files_test[i].write(line)

                start = end

    # 关闭所有文件
    for f in files_train + files_test:
        f.close()

    print(f'{k}-fold 数据列表生成完毕。')

# 调用函数
data_path = '/home/aistudio/data/data23668/Dataset'
split_data_k_fold(data_path, k=5)
import math

class Neuron:
    def __init__(self, bias):
        self.bias = bias
        self.weights = []
        self.inputs = []
        self.output = None

    def calculate_output(self, inputs):
        self.inputs = inputs
        if len(self.weights) != len(inputs):
            raise ValueError(f"权重数量 ({len(self.weights)}) 与输入数量 ({len(inputs)}) 不匹配")
        total_input = self.calculate_total_net_input()
        self.output = self.squash(total_input)
        return self.output

    def calculate_total_net_input(self):
        #print(f"Inputs: {self.inputs}")
        #print(f"Weights: {self.weights}")
        return sum(input * weight for input, weight in zip(self.inputs, self.weights)) + self.bias

    @staticmethod
    def squash(total_net_input):
        return 1.0 / (1.0 + math.exp(-total_net_input))

    def calculate_pd_error_wrt_total_net_input(self, target_output):
        return self.calculate_pd_error_wrt_output(target_output) * self.calculate_pd_total_net_input_wrt_input()

    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    def calculate_pd_error_wrt_output(self, target_output):
        return -(target_output - self.output)

    def calculate_pd_total_net_input_wrt_input(self):
        return self.output * (1 - self.output)

    def calculate_pd_total_net_input_wrt_weight(self, index):
        return self.inputs[index]
import random

class NeuronLayer:
    def __init__(self, num_neurons, num_inputs, bias=None, weights=None):
        self.bias = bias if bias is not None else random.random()
        self.num_inputs = num_inputs  # 每个神经元的输入连接数

        self.neurons = []
        for i in range(num_neurons):
            new_neuron = Neuron(self.bias)
            self.init_weights(new_neuron, weights)
            self.neurons.append(new_neuron)

    def init_weights(self, neuron, weights=None):
        if weights:
            neuron.weights = weights
        else:
            neuron.weights = [random.random() for _ in range(self.num_inputs)]

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def train(self, output_errors, learning_rate):
        pd_errors_wrt_output = [neuron.calculate_pd_error_wrt_total_net_input(error)
                                for neuron, error in zip(self.neurons, output_errors)]

        for i, neuron in enumerate(self.neurons):
            for j in range(len(neuron.weights)):
                pd_error_wrt_weight = pd_errors_wrt_output[i] * neuron.inputs[j]
                neuron.weights[j] -= learning_rate * pd_error_wrt_weight

        pd_errors_wrt_input_layer = [0] * self.num_inputs
        for i in range(self.num_inputs):
            for neuron, pd_error in zip(self.neurons, pd_errors_wrt_output):
                pd_errors_wrt_input_layer[i] += pd_error * neuron.weights[i]

        return pd_errors_wrt_input_layer


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers_info, output_size):
        self.layers = []
        previous_inputs = input_size  # 第一层的输入数量是输入数据的特征数量

        # 动态创建隐藏层
        for num_neurons in hidden_layers_info:
            self.layers.append(NeuronLayer(num_neurons, previous_inputs))
            previous_inputs = num_neurons  # 更新输入数量为当前层的输出

        # 添加输出层
        self.layers.append(NeuronLayer(output_size, previous_inputs))

    def feed_forward(self, inputs):
        for layer in self.layers:
            inputs = layer.feed_forward(inputs)
        return inputs

    def train(self, training_data, labels, learning_rate, epochs, batch_size=32):
        for epoch in range(epochs):
            total_loss = 0
            correct_predictions = 0
            total_samples = 0

            for batch_id in range(0, len(training_data), batch_size):
                batch_data = training_data[batch_id:batch_id + batch_size]
                batch_labels = labels[batch_id:batch_id + batch_size]

                batch_loss = 0
                batch_correct = 0

                for inputs, label in zip(batch_data, batch_labels):
                    outputs = self.feed_forward(inputs)
                    target_outputs = [0] * len(outputs)
                    target_outputs[label] = 1

                    errors = [target - output for target, output in zip(target_outputs, outputs)]
                    loss = np.sum(np.square(errors)) / 2
                    batch_loss += loss
                    total_loss += loss

                    if np.argmax(outputs) == label:
                        batch_correct += 1
                        correct_predictions += 1

                    for i in reversed(range(len(self.layers))):
                        errors = self.layers[i].train(errors, learning_rate)

                total_samples += len(batch_data)
                train_loss = batch_loss / len(batch_data)
                train_acc = batch_correct / len(batch_data)

                print(f"train_pass:{epoch}, batch_id:{batch_id}, train_loss:[{train_loss}], train_acc:[{train_acc}]")

            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / total_samples}, Accuracy: {correct_predictions / total_samples}")

    def predict(self, inputs):
        return self.feed_forward(inputs)
# 数据预处理函数
def data_mapper(sample):
    img_path, label = sample.split('\t')
    img = cv2.imread(img_path)  # 加载图像
    img = cv2.resize(img, (100, 100))  # 调整图像大小
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
    img = img.flatten()  # 将图像展平为一维数组
    img = img.astype('float32') / 255.0  # 归一化
    print(f"Processed image shape: {img.shape}")
    return img, int(label)

# 数据读取器
def data_reader(data_list_paths, num_workers=16):
    def reader():
        samples = []
        for data_list_path in data_list_paths:
            with open(data_list_path, 'r') as f:
                lines = f.readlines()
                samples.extend([line.strip() for line in lines])

        with Pool(num_workers) as p:
            processed_samples = p.map(data_mapper, samples)

        for sample in processed_samples:
            yield sample

    return reader

# 训练神经网络函数
def train_neural_network():
    train_data_path = 'train_data_fold_1.txt'  # 你的训练数据文件路径
    train_reader_func = data_reader([train_data_path], num_workers=2)  # 获取读取器函数
    train_reader = list(train_reader_func())  # 调用读取器函数并转换为列表

    training_data = [sample[0] for sample in train_reader]
    labels = [sample[1] for sample in train_reader]

    input_size = 100 * 100 * 3  # 例如，对于100x100的RGB图像
    hidden_layers_info = [100, 50]  # 定义每层的神经元数
    output_size = 10  # 假设有10个类别

    # 实例化模型
    network = NeuralNetwork(input_size, hidden_layers_info, output_size)

    # 设置学习率和训练轮数
    learning_rate = 0.01
    epochs = 10

    # 开始训练
    network.train(training_data, labels, learning_rate, epochs, batch_size=32)

train_neural_network()
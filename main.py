import numpy  # Бибилотека, содержащая функции, позволяющие реализовывать операции высшей математики (матрицы)
import scipy.special  # Данная библиотека содержит функцию сигмоиды
import matplotlib.pyplot  # Библиотека с возможностью графического построения


# Определение класса нейронной сети
class NeuralNetwork:

    # Инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Задается количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Коэффициент обучения
        self.lr = learningrate

        # Матрицы весовых коэффициентов связей wih (между входным и скрытым слоями)
        # и who (между скрытым и выходным слоями).
        # Весовые коэффициенты для связей между узлом i и узлом j следующего слоя
        # обозначены как w_i_j:
        # w11 w21
        # w12 w22 и т.д.
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Определение анонимной функции.
        # Здесь сигмоида используется в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # Тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # Преобразование списка входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # Расчет входящих и исходящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Расчет входящих и исходящих сигналов для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Расчет ошибки как: ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # Ошибки скрытого слоя считаются как ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Обновление весовых коэффициентов связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))

        # Обновление весовых коэффициентов связей между входным и скрытым слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    # Опрос нейронной сети
    def query(self, inputs_list):
        # Преобразование списка входных значений
        # в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # Расчет входящих и исходящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Расчет входящих и исходящих сигналов для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


# Инициируем количество входных, скрытых и выходных переменных
input_nodes = 784  # В данном случае число обосновано количеством анализируемых пикселей (28*28)
hidden_nodes = 100  # Не имеет строгого обоснования, выявляется подбором до достижения желаемого результата
output_nodes = 10  # Обосновано максимальным количеством возможных вариантов ответа (цифр всего 10)

# Коэффициент обучения
learning_rate = 0.3

# Создаем экземпляр нейронной сети
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Загрузка в список тестового набора данных CSV-файла из набора MNIST
training_data_file = open("mnist_dataset/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# Тренировка нейронной сети

# Перебор всех записей в тренировочном наборе данных
for record in training_data_list:
    # Получаем список значений, используя в качестве разделителя (",")
    all_values = record.split(',')
    # Масштабируем и смещаем входные значения
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # Создаем целевые выходные значения (все равны 0.01, за исключением маркерного значения, равного 0.99
    targets = numpy.zeros(output_nodes) + 0.01
    # all_values[0] - целевое маркерное значение для данной записи
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    pass

# Получаем из CSV-файла тестовые данные в строках
test_data_file = open("mnist_dataset/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# Тестирование нейронной сети

# Журнал оценок работы сети, изначально пустой массив
scorecard = []

# Код, реализующий перебор всех записей в тестовом наборе данных
for record in test_data_list:
    # Получаем список значений из записи, используя символы запятой в качестве разделителя
    all_values = record.split(',')
    # Правильный ответ - первое значение
    correct_label = int(all_values[0])
    print(correct_label, " истинный маркер")
    # Масштабирование и смещение входных значение
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # Опрос сети
    outputs = n.query(inputs)
    # Индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
    print(label, " ответ сети")
    # Занесение оценки ответа сети к концу журнала работы сети
    if (label == correct_label):
        # В случае правильного ответа сети в список заносится значение 1
        scorecard.append(1)
    else:
        # В случае неверного ответа в список заносится значение 0
        scorecard.append(0)
        pass
    pass

# Вывод массива журнала
print(scorecard)

# Рассчет показателя эффективности в виде доли правильных ответов
scorecard_array = numpy.asarray(scorecard)
print("Эффективность - ", scorecard_array.sum() / scorecard_array.size)

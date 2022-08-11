# код рассчитан на python 3.7 и tensorflow 1.15.0

import datetime
import warnings
from collections import defaultdict
from copy import deepcopy

import joblib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.training.optimizer import Optimizer

warnings.filterwarnings("ignore", category=DeprecationWarning)


def _weight_variable(shape):
    # для весов полносвязного слоя инициализируем значения по Каймину Ге (Хе)
    stddev = 2. / np.sqrt(shape[0])
    initial = tf.random.truncated_normal(shape=shape, mean=0.0, stddev=stddev)
    return tf.Variable(initial, dtype=tf.float32)


def _bias_variable(shape):
    # смещения инициализируем нулями
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)


def initialize_uninitialized_global_variables(sess):
    """
    Only initializes the variables of a TensorFlow session that were not
    already initialized.
    :param sess: the TensorFlow session
    :return:
    """
    # List all global variables
    global_vars = tf.global_variables()

    # Find initialized status for all variables
    is_var_init = [tf.is_variable_initialized(var) for var in global_vars]
    is_initialized = sess.run(is_var_init)

    # List all variables that were not initialized previously
    not_initialized_vars = [var for (var, init) in
                            zip(global_vars, is_initialized) if not init]

    # Initialize all uninitialized variables found, if any
    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


class _WVA_SGD(tf.train.GradientDescentOptimizer):

    def __init__(self, learning_rate, use_locking=False, name="GradientDescent"):
        super(_WVA_SGD, self).__init__(learning_rate, use_locking, name)

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None, impacts=None):
        """ comments """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))

        if impacts is None:
            processed_grads_and_vars = grads_and_vars
        else:
            impact = iter(impacts)
            processed_grads_and_vars = []
            for g, v in grads_and_vars:
                if g is None:
                    processed_grads_and_vars.append((g, v))
                else:
                    processed_grads_and_vars.append((tf.multiply(g, next(impact)), v))

        return self.apply_gradients(processed_grads_and_vars,
                                    global_step=global_step,
                                    name=name)


class _WVA_Adam(tf.train.AdamOptimizer):

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 use_locking=False,
                 name="Adam"):
        super(_WVA_Adam, self).__init__(learning_rate, beta1, beta2, epsilon, use_locking, name)

    def minimize(self, loss, global_step=None, var_list=None,
                 gate_gradients=Optimizer.GATE_OP, aggregation_method=None,
                 colocate_gradients_with_ops=False, name=None,
                 grad_loss=None, impacts=None):
        """ comments """
        grads_and_vars = self.compute_gradients(
            loss, var_list=var_list, gate_gradients=gate_gradients,
            aggregation_method=aggregation_method,
            colocate_gradients_with_ops=colocate_gradients_with_ops,
            grad_loss=grad_loss)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
            raise ValueError(
                "No gradients provided for any variable, check your graph for ops"
                " that do not support gradients, between variables %s and loss %s." %
                ([str(v) for _, v in grads_and_vars], loss))

        if impacts is None:
            processed_grads_and_vars = grads_and_vars
        else:
            impact = iter(impacts)
            processed_grads_and_vars = []
            for g, v in grads_and_vars:
                if g is None:
                    processed_grads_and_vars.append((g, v))
                else:
                    processed_grads_and_vars.append((tf.multiply(g, next(impact)), v))

        return self.apply_gradients(processed_grads_and_vars,
                                    global_step=global_step,
                                    name=name)


class Model:
    def __init__(self, shape, session, learning_rate=0.01, impact_to_grads=False):
        """
        :param shape:   структура сети - список из чисел нейронов в каждом слое сети
                        от входа к выходу справа налево, например, [784, 100, 10]
        :param session: tensorflow-сессия для расчетов сети
        """

        self.session = session
        self._shape = shape
        depth = len(shape) - 1
        if depth < 1:
            raise ValueError("Недопустимая структура сети!")

        # заглушки для входных данных
        self.x = tf.placeholder(tf.float32, shape=[None, shape[0]])
        self.labels = tf.placeholder(tf.float32, shape=[None, shape[-1]])

        # все веса слоев сети будем хранить в списке
        self.var_list = []
        self.vars_shadow = []
        self.impacts = []
        for ins, outs in zip(shape[:-1], shape[1:]):
            self.var_list.append(_weight_variable([ins, outs]))
            self.var_list.append(_bias_variable([outs]))
            self.vars_shadow.append(_bias_variable([ins, outs]))
            self.vars_shadow.append(_bias_variable([outs]))
            self.impacts.append(_bias_variable([ins, outs]))
            self.impacts.append(_bias_variable([outs]))

        # список для хранения важностей весов сети
        self.wb_importance = [np.zeros(v.shape, dtype=np.float32) for v in self.var_list]

        # строим вычислительный граф
        outputs = []
        x, y, z = self.x, None, None
        for i in range(depth):
            z = tf.matmul(x, self.var_list[i * 2]) + self.var_list[i * 2 + 1]
            y = tf.nn.softmax(z) if i == depth-1 else tf.nn.leaky_relu(z)
            outputs.append(y)
            x = y

        # функция стоимости
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=self.labels))

        # точность (accuracy)
        self.correct_preds = tf.equal(tf.argmax(z, axis=1), tf.argmax(self.labels, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_preds, tf.float32))

        # вычисляем суммарный по модулю прошедший сигнал
        self.signals = []

        os = tf.reduce_mean(tf.abs(self.x), axis=0)
        for i in range(depth):
            ws = tf.transpose(tf.multiply(os, tf.transpose(tf.abs(self.var_list[i*2]))))
            self.signals.append(ws)
            os = tf.reduce_mean(tf.abs(outputs[i]), axis=0)
            self.signals.append(os)

        self._train_step = None
        self._store_shadow = [tf.assign(s, v) for v, s in zip(self.var_list, self.vars_shadow)]
        self._correct_vars = [tf.assign(v, s + i * (v - s))
                              for v, s, i in zip(self.var_list, self.vars_shadow, self.impacts)]

        self.reset()
        self._impact_to_grads = impact_to_grads

        # устанавливаем шаг оптимизатора
        if impact_to_grads:
            self._train_step = _WVA_SGD(learning_rate).minimize(self.loss, impacts=self.impacts)
            # self._train_step = _WVA_Adam(learning_rate).minimize(self.loss, impacts=self.impacts)
        else:
            self._train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
            # self._train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        initialize_uninitialized_global_variables(self.session)

    def reset(self):
        # инициализируем веса сети
        for v, s, i in zip(self.var_list, self.vars_shadow, self.impacts):
            self.session.run(v.initializer)
            self.session.run(tf.assign(s, v))
            self.session.run(i.initializer)

        # список для хранения важностей весов сети
        for i in self.wb_importance:
            i.fill(0.)

    def open_lesson(self, lmbda=0.0):
        """
        Открытие урока обучения сети на отдельном датасете
        :param learning_rate: скорость обучения для SGD
        :param lmbda:         коэффициент влияния важностей - насколько сильно
                              важности тянут веса к эталонным значениям
        """
        for impact, imp in zip(self.impacts, self.wb_importance):
            impact.load(1. / (1. + lmbda * imp))

    def train_step(self, train_batch):
        if not self._impact_to_grads:
            self.session.run(self._store_shadow)

        feed_dict = {self.x: train_batch[0], self.labels: train_batch[1]}
        self._train_step.run(feed_dict=feed_dict)

        if not self._impact_to_grads:
            self.session.run(self._correct_vars)

    def close_lesson(self, closing_set=None):
        """
        Закрытие урока обучения сети на отдельном датасете. Расчет и накопление важностей весов.
        :param closing_set: датасет, на котором будут рассчитаны важности весов после обучения
        :return:
        """

        # рассчитываем важности весов на закрывающем датасете
        addendum = self.session.run(self.signals, feed_dict={self.x: closing_set})

        # добавляем рассчитанные важности к сохраненным
        for i, a in zip(self.wb_importance, addendum):
            i += a


# функция случайным образом переставляет входы одинаково для всех примеров датасета
def permute_mnist(mnist):
    perm_inds = list(range(mnist.train.images.shape[1]))
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name)
        this_set._images = np.transpose(np.array([this_set.images[:, c] for c in perm_inds]))
    return mnist2


def train_model(model, train_set, test_sets, batch_size=100, epochs=1):
    """
    Обучение модели
    :param model:       обучаемая модель
    :param train_set:   обучающий датасет
    :param test_sets:   список датасетов, на которых будет считаться средняя точность
    :param batch_size:  размер батча
    :param epochs:      количество эпох обучения
    :return:            средняя точность на тестовых датасетах после обучения
    """
    num_iters = int(np.ceil(len(train_set.train.labels) * epochs / batch_size))
    for idx in range(num_iters):
        train_batch = train_set.train.next_batch(batch_size)
        model.train_step(train_batch)
        if idx % 67 == 0:
            print(f'\rTraining  {idx + 1}/{num_iters} iterations done.', end='')

    print(f'\rTraining  {num_iters}/{num_iters} iterations done. ', end='')

    accuracy = 0.
    for t, test_set in enumerate(test_sets):
        feed_dict = {model.x: test_set.test.images, model.labels: test_set.test.labels}
        accuracy += model.accuracy.eval(feed_dict=feed_dict)
    accuracy /= len(test_sets)
    print(f'Mean accuracy on {len(test_sets)} test sets is {accuracy}')
    return accuracy


def continual_learning(model, data_sets, lmbda):
    """
    Последовательное обучение на нескольких обучающих наборах
    :param net_struct: структура сети
    :param data_sets:  список обучающих датасетов для последовательного обучения
    :param session:    tf-сессия
    :param lr:         скорость обучения
    :param lmbda:      степень влияния важностей на обучение
    :return:           список усредненных по выученным датасетам оценок
    """
    model.reset()
    print('Model has been cleaned.')
    test_sets = []
    accuracies = []
    for data_set in data_sets:
        test_sets.append(data_set)
        model.open_lesson(lmbda)
        accuracy = train_model(model, data_set, test_sets, 100, 4)
        accuracies.append(accuracy)
        model.close_lesson(data_set.validation.images)
    return accuracies

# создаем tf-сессию
sess = tf.InteractiveSession()


# пробуем загрузить обучающие наборы
dataset_file = 'datasets.dmp'
try:
    data_sets = joblib.load(dataset_file)
    print('Dataset has been loaded from cache.')
except FileNotFoundError:
    print('Dataset cache not found. Creating new one.')
    # считываем данные MNIST
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # создаем 10 различных обучающих наборов для последовательного обучения
    mnist0 = mnist
    mnist1 = permute_mnist(mnist)
    mnist2 = permute_mnist(mnist)
    mnist3 = permute_mnist(mnist)
    mnist4 = permute_mnist(mnist)
    mnist5 = permute_mnist(mnist)
    mnist6 = permute_mnist(mnist)
    mnist7 = permute_mnist(mnist)
    mnist8 = permute_mnist(mnist)
    mnist9 = permute_mnist(mnist)

    data_sets = [mnist0, mnist1, mnist2, mnist3, mnist4, mnist5, mnist6, mnist7, mnist8, mnist9]
    joblib.dump(data_sets, dataset_file, compress=3)


exp_file = 'wva-sgd-1x-deltas.dmp'
try:
    experiments = joblib.load(exp_file)
except FileNotFoundError:
    print('Experiment cache not found. Creating new one.')
    experiments = defaultdict(list)


# определим параметры обучения
net_struct = [784, 300, 150, 10]
learning_rate = 0.2  # 0.001
N = 10

model = Model(net_struct, sess, learning_rate, False)

start_time = datetime.datetime.now()
time_format = "%Y-%m-%d %H:%M:%S"
print(f'Continual learning start at {start_time:{time_format}}')

for lmbda in np.arange(50., 5000., 200.):
    exps = experiments[lmbda]
    len_exp = len(exps)
    K = max(0, N - len_exp)
    print(f'Start calc on lambda {lmbda}. {K} experiments are queued.')
    for i in range(K):
        iter_start_time = datetime.datetime.now()
        print(f'{i+1+len_exp}-th experiment on lambda {lmbda} started at {iter_start_time:{time_format}}')
        accuracies = continual_learning(model, data_sets, lmbda=lmbda)
        exps.append(accuracies)
        joblib.dump(experiments, exp_file)
        print(f'{i+1}-th experiment time spent {datetime.datetime.now() - iter_start_time}')
        print(f'For now total time spent {datetime.datetime.now() - start_time}')

dataset_num = range(1, len(accuracies) + 1)

# нарисуем график деградации средней точности на всех выученных датасетах
plt.figure(figsize=(7, 3.5))
plt.ylim(0.09, 1.)
plt.xlim(1, len(accuracies))
plt.ylabel('Total accuracy')
plt.xlabel('Number of tasks')
plt.plot(dataset_num, accuracies, marker=".")
#plt.legend()
plt.show()

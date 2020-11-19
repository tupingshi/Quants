import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import pandas as pd
from sklearn.preprocessing import Imputer  #处理缺失值
import os
os.chdir('C:\\Users\\Pingshi Tu\\Desktop\\python')


#--------------------------------导入数据--------------------------------------
f = open('dataset_2.csv')
df = pd.read_csv(f)     #读入股票数据
data = df.iloc[:,2:-1].values  #取第3-9列


#-------------------------------初步处理数据-----------------------------------
class Process_Data():
    #------------------------------------------------------ 1.转换成有监督的模式
    def series_to_supervised(self, data, n_in=1, n_out=1): #n_in,n_out相当于lag
        n_vars = data.shape[1] #变量个数
        df = pd.DataFrame(data)
        cols, names = list(), list()
        #输入序列(t-n. ... , t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)'%(j+1,i)) for j in range(n_vars)]        
        #预测序列(t, t+1, ... , t+n)  
        for i in range(0, n_out):     
            cols.append(df.shift(-i)) 
            if i == 0:#t时刻
                names += [('var%d(t)'%(j+1)) for j in range(n_vars)]
            else: 
                names += [('var%d(t+%d)'%(j+1, i)) for j in range(n_vars)]
        #拼接 
        agg = pd.concat(cols, axis=1)    
        agg.columns = names        
        #将空值NaN行删除
        agg.dropna(inplace = True)
        agg = np.array(agg)
        agg = agg[:, 0:(data.shape[1] + 1)]
        return agg
    #--------------------------------------------------------------- 2.特征工程
    def feature_engineering(self, data):
        mean = np.nanmean(data, 0)
        std = np.nanstd(data, 0)
        standardized_data = (data - mean) / std
        if np.isnan(standardized_data.all()): #缺失值计算，参数strategy为缺失值填充方式，默认为mean（均值）  
            standardized_data = Imputer().fit_transform(standardized_data)
        return mean, std, standardized_data
    #------------------------------------------------------------- 3.获取训练集
    def get_train_data(self, standardized_data, batch_size = 60, time_step = 20, train_begin = 0, train_end = 5800):
        batch_index = []
        data_train = standardized_data[train_begin: train_end]
        train_x, train_y=[], []   #训练集
        for i in range(len(data_train) - time_step):
           if i % batch_size == 0:
               batch_index.append(i)
           x = data_train[i:i+time_step, :(data.shape[1]-1)]
           y = data_train[i:i+time_step, -1, np.newaxis]
           train_x.append(x.tolist())
           train_y.append(y.tolist())
        batch_index.append((len(data_train) - time_step))
        return batch_index, train_x, train_y
    #------------------------------------------------------------- 4.获取测试集
    def get_test_data(self, standardized_data, time_step = 20, test_begin = 5800):
        data_test = standardized_data[test_begin:]
        size = (len(data_test) + time_step - 1) // time_step  #有size个sample
        test_x, test_y = [], []
        for i in range(size - 1):
           x = data_test[i*time_step: (i+1)*time_step, :(data.shape[1]-1)]
           y = data_test[i*time_step: (i+1)*time_step, -1]
           test_x.append(x.tolist())
           test_y.extend(y)
        test_x.append((data_test[(i+1)*time_step:, :(data.shape[1]-1)]).tolist())
        test_y.extend((data_test[(i+1)*time_step:, -1]).tolist())
        return test_x, test_y

#预处理数据
Process_Data = Process_Data()
#转换成有监督的形式
data = Process_Data.series_to_supervised(data)
#特征工程
mean, std, standardized_data = Process_Data.feature_engineering(data)
#获取训练集
batch_index, train_x, train_y = Process_Data.get_train_data(standardized_data)
#获取测试集
test_x, test_y = Process_Data.get_test_data(standardized_data)


#-----------------------LSTM模型的定义、训练、预测------------------------------
class Define_Train_Predict_LSTM():
    #--------------------------------------------------------------- 1.定义变量
    def __init__(self):
        self.rnn_unit = rnn_unit          #隐层神经元的个数
        self.lstm_layers = lstm_layers       #隐层层数
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr     #学习率
        self.time_step = time_step
        self.weights = weights
        self.biases = biases
        self.keep_prob = keep_prob
        self.X = X
        self.Y = Y
    #------------------------------------------------------- 2.定义一个lstm模型
    def define_lstm(self, X):
        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = weights['in']
        b_in = biases['in']
        #将tensor转成2维进行计算，进行矩阵运算，计算后的结果作为隐藏层的输入
        input = tf.reshape(X, [-1, input_size])  
        #input_rnn = tf.nn.tanh(tf.matmul(input, w_in) + b_in)
        input_rnn = tf.matmul(input, w_in) + b_in
        #将tensor转成3维，作为lstm cell的输入
        input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  
        #定义一个lstm神经元
        def lstmCell():
            #basicLstm单元
            basicLstm = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
            # dropout
            drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, 
                                                 output_keep_prob = keep_prob)
            return basicLstm
        #叠加多层神经网络
        cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(lstm_layers)])
        init_state = cell.zero_state(batch_size, dtype=tf.float32)
        output_rnn, final_states = tf.nn.dynamic_rnn(cell, 
                                                     input_rnn, 
                                                     initial_state = init_state, 
                                                     dtype = tf.float32)
        #输出层
        output = tf.reshape(output_rnn, [-1, rnn_unit]) 
        w_out = weights['out']
        b_out = biases['out']
        #输出预测值
        pred = tf.matmul(output, w_out) + b_out
        #pred = tf.nn.tanh(tf.matmul(output, w_out) + b_out)
        return pred, final_states
    #--------------------------------------------------------- 3.训练、预测lstm
    def train_predict_lstm(self):
        global test_y
        #定义损失函数
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
        #定义Adam优化器，将损失最小化
        train_op = tf.train.AdamOptimizer(lr).minimize(loss)
        #训练lstm模型
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 15)
        #初始化运行加载各个变量
        init = tf.global_variables_initializer()
        with tf.Session() as sess:            
            try:
                #定义了存储模型的文件路径
                module_file = tf.train.latest_checkpoint('C:\\Users\\Pingshi Tu\\Desktop\\python\\model_save')
                saver.restore(sess, module_file)
                print ('成功加载模型参数')
            except:
                #如果是第一次运行，通过init告知tf加载并初始化变量
                print ('未加载模型参数，文件被删除或者第一次运行')
                sess.run(init)
            for i in range(100):     #这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
                for step in range(len(batch_index) - 1):
                    _, loss_, = sess.run([train_op, loss],
                                          feed_dict = {X: train_x[batch_index[step]: batch_index[step + 1]],
                                                       Y: train_y[batch_index[step]: batch_index[step + 1]], 
                                                       keep_prob: 0.2})            
                if (i+1) % 5 == 0:
                    print('迭代次数:', i+1, ' loss:', loss_)
            print('完成训练，开始模型预测，并计算预测精确度...')
            #开始模型预测，并计算预测精确度
            test_predict = []
            for step in range(len(test_x) - 1):
              prob = sess.run(pred, feed_dict = {X: [test_x[step]], keep_prob: 1})
              predict = prob.reshape((-1))
              test_predict.extend(predict)
            test_y = np.array(test_y) * std[7] + mean[7]
            test_predict = np.array(test_predict) * std[7] + mean[7]
            accuracy = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  #偏差程度
            print("预测精度:", accuracy)
            #以折线图表示结果
            plt.figure()
            plt.plot(list(range(len(test_predict))), test_predict, color='b',)
            plt.plot(list(range(len(test_y))), test_y,  color='r')
            plt.show()    
            print('保存模型:', saver.save(sess, 'C:\\Users\\Pingshi Tu\\Desktop\\python\\model_save\\modle.ckpt'))
            print('完成模型的定义、训练、及预测过程')
        return accuracy, test_predict


rnn_unit = 10        #隐层神经元的个数
lstm_layers = 2       #隐层层数
input_size = 7
output_size = 1
lr = 0.0006         #学习率
time_step = 20
weights = {'in': tf.Variable(tf.random_normal([input_size, rnn_unit])), 'out': tf.Variable(tf.random_normal([rnn_unit, 1]))}
biases = {'in': tf.Variable(tf.constant(0.1, shape = [rnn_unit, ])), 'out': tf.Variable(tf.constant(0.1, shape = [1, ]))}
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
X = tf.placeholder(tf.float32, shape = [None, time_step, input_size])
Y = tf.placeholder(tf.float32, shape = [None, time_step, output_size])

Define_Train_Predict_LSTM = Define_Train_Predict_LSTM()
with tf.variable_scope('sec_lstm'):
    pred, final_states = Define_Train_Predict_LSTM.define_lstm(X)
accuracy, test_predict = Define_Train_Predict_LSTM.train_predict_lstm()

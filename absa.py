
import numpy as np
from collections import Counter
import tensorflow as tf
import os
from sklearn.metrics import precision_recall_fscore_support as score
np.set_printoptions(threshold=np.inf)

def batch_index(length, batch_size, n_iter=100, is_shuffle=True):
    index = list(range(length))
    for j in range(n_iter):
        if is_shuffle:
            np.random.seed()
            np.random.shuffle(index)
        if length%batch_size:
            num_batch = int(length/batch_size) + 1
        else:
            num_batch  = int(length/batch_size)
        for i in range(num_batch):
            yield index[i*batch_size:(i + 1)*batch_size]

def change_y_to_onehot(y):
    ## 0 :- neutral , 1 :- positive , -1 :- negative
    ## Label matrix which contain one-hot vector corresponding to each label y
    print("Data distribution :-")
    print(Counter(y))
    class_set = set(y)
    n_class = len(class_set)
    y_onehot_mapping = dict(zip(class_set, range(n_class))) ## map (0,1,-1) -> (0,1,2) where 0,1,2 are the indexes of each onehot vector
    onehot = []
    for label in y:
        tmp = [0] * n_class
        tmp[y_onehot_mapping[label]] = 1
        onehot.append(tmp)
    return np.asarray(onehot, dtype=np.int32)


def load_word_id_mapping(word_id_file):
    word_to_id = dict()
    for line in open(word_id_file):
        #line = line.decode(encoding, 'ignore').lower().split()
        line = line.lower().split()
        word_to_id[line[0]] = int(line[1])
    print('\nload word-id mapping done!\n')
    return word_to_id

def load_w2v(w2v_file, embedding_dim):
    fp = open(w2v_file)
    #if is_skip:
        #fp.readline()
    w2v = []
    word_dict = dict()
    # [0,0,...,0] represent absent words
    w2v.append([0.] * embedding_dim)
    cnt = 0
    for line in fp:
        cnt += 1
        line = line.split()
        if len(line) != embedding_dim + 1:
            print('a bad word embedding: {}'.format(line[0]))
            continue
        w2v.append([v for v in line[1:]])
        word_dict[line[0]] = cnt
    w2v = np.asarray(w2v, dtype=np.float32)
    w2v = np.row_stack((w2v, np.sum(w2v, axis=0) / cnt)) ##appending row wise average of all word vectors
    print(np.shape(w2v))
    word_dict['$t$'] = (cnt + 1) ##denote $t$ to special words that are not present in glove and in data
    print(word_dict['$t$'], len(w2v))
    return word_dict, w2v

def load_word_embedding(word_id_file, w2v_file, embedding_dim):
    ##word_id_file :- txt file contains word with corresponding ids
    ##w2v_file :- txt file contains words with corresponsing vectors
    ##In this we have word_id_file as word_id_mapping of whole corpus [train + test]
    ## w2v_file is word vector file in which all words of glove twitter corpus have their vectors.
    word_to_id = load_word_id_mapping(word_id_file)
    word_dict, w2v = load_w2v(w2v_file, embedding_dim)
    cnt = len(w2v)
    ##updating w2v with word vectors of those words which are not present in w2v_file but present in word_id_file
    for k in word_to_id.keys():
        if k not in word_dict:
            word_dict[k] = cnt
            w2v = np.row_stack((w2v, np.random.uniform(-0.01, 0.01, (embedding_dim,))))
            cnt += 1
    print(len(word_dict), len(w2v))
    return word_dict, w2v

def load_aspect2id(input_file, word_id_mapping, w2v, embedding_dim):
    ##input_file :- aspect_id_file [txt file of aspects with corresponding ids]
    ##word_id_mapping :- dictionary mapping word with ids
    ##w2v :- word vector file (txt file contains words with corresponsing vectors)
    ##embedding_dim :- 300
    aspect2id = dict()
    a2v = list()
    a2v.append([0.] * embedding_dim)
    cnt = 0
    for line in open(input_file):
        line = line.lower().split()
        cnt += 1
        aspect2id[' '.join(line[:-1])] = cnt   ##aspect can be multi-word (more than 1 more together can form an aspect)
        tmp = []
        for word in line:
            if word in word_id_mapping:
                tmp.append(w2v[word_id_mapping[word]])
        if tmp: ## if there is any word embedding for aspects words
            a2v.append(np.sum(tmp, axis=0) / len(tmp)) ## if aspect term contains n words then word vector for that aspect is average of n words vector
        else: ## is aspect words don't have word embedding then randomly intialise a vector for it
            a2v.append(np.random.uniform(-0.01, 0.01, (embedding_dim,)))
    print(len(aspect2id), len(a2v))
    return aspect2id, np.asarray(a2v, dtype=np.float32)


def load_inputs_at(input_file, word_id_file, aspect_id_file, sentence_len):
    ## Here input_file is training/testing data [a corpus containing sentences and corresponding aspects with polarity]
    ## word_id_file is dictionary of word to id mapped
    ## aspect_id_file is dictionary of aspect to id mapped
    ## sentence_len is maximum sentence length allowable i.e, 80

    ##Returns :- 1. x, data matrix where each row represents sentence vector of length 80
    ## 2. y, label matrix of one-hot vector of each label (0,1,-1)
    ## 3. aspect_words , list of all ids of all aspect terms of data
    ## 4. sen_len , length of all sentences of data


    #if type(word_id_file) is not str:
    word_to_id = word_id_file
    print('load word-to-id done!')
    #if type(aspect_id_file) is not str:
    aspect_to_id = aspect_id_file
    print('load aspect-to-id done!')

    x, y, sen_len = [], [], []
    aspect_words = [] # contains aspect term id and if aspect term is not present in mapping dictionary (aspect_to_id) then it store 0
    lines = open(input_file).readlines()
    for i in range(0, len(lines), 3):
        aspect_word = ' '.join(lines[i + 1].lower().split())
        aspect_words.append(aspect_to_id.get(aspect_word, 0))

        y.append(lines[i + 2].split()[0])
        #words = lines[i].decode(encoding).lower().split()
        words = lines[i].lower().split()
        ids = []
        for word in words:
            if word in word_to_id:
                ids.append(word_to_id[word])
        # ids = list(map(lambda word: word_to_id.get(word, 0), words))
        sen_len.append(len(ids))
        x.append(ids + [0] * (sentence_len - len(ids))) ## [0]*(sentence_len - len(ids) is extra padding in order to make each sentence vector of equal size i,e. 80
    cnt = 0
    for item in aspect_words:
        if item > 0:
            cnt += 1
    print('cnt=', cnt) ##total aspect words of corpus that are present in aspect_to_id dictionary
    y = change_y_to_onehot(y)
    for item in x: ##checking if any sentence vector is of length <80
        if len(item) != sentence_len:
            print('aaaaa=', len(item))
    x = np.asarray(x, dtype=np.int32)
    return x, np.asarray(sen_len), np.asarray(aspect_words), np.asarray(y)

class LSTM(object):

    def __init__(self, embedding_dim=100, batch_size=64, n_hidden=100, learning_rate=0.01,n_class=3, max_sentence_len=80, l2_reg=0., n_iter=25, model_type="!bi", model_name="AT"):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_class = n_class
        self.max_sentence_len = max_sentence_len
        self.l2_reg = l2_reg
        self.n_iter = n_iter
        self.model_type = model_type
        self.model_name = model_name
        self.word_id_mapping, self.w2v = load_word_embedding(FLAGS.word_id_file_path, FLAGS.embedding_file_path, self.embedding_dim)
        self.word_embedding = tf.Variable(self.w2v, dtype=tf.float32, name='word_embedding')
        self.aspect_id_mapping, self.aspect_embed = load_aspect2id(FLAGS.aspect_id_file_path, self.word_id_mapping, self.w2v, self.embedding_dim)
        self.aspect_embedding = tf.Variable(self.aspect_embed, dtype=tf.float32, name='aspect_embedding')
        self.alpha = tf.placeholder(tf.float32,[None,self.max_sentence_len],name='alpha')
        with tf.name_scope('inputs'):
            self.x = tf.placeholder(tf.int32, [None, self.max_sentence_len], name='x') ## batch_size * 80 (25 * 80)
            self.y = tf.placeholder(tf.int32, [None, self.n_class], name='y') ## batch_size * 3 (25* 3)
            self.sen_len = tf.placeholder(tf.int32, None, name='sen_len') ## list of lengths of all sentences of a batch (list of length 25 )
            self.aspect_id = tf.placeholder(tf.int32, None, name='aspect_id') ##list of ids of all aspect terms of a batch (list of length 25 )

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.get_variable(
                    name='softmax_w',
                    shape=[self.n_hidden, self.n_class], ## 300 * 3
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)

                )
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.get_variable(
                    name='softmax_b',
                    shape=[self.n_class], ## 3*1
                    initializer=tf.random_uniform_initializer(-0.01, 0.01),
                    regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)

                )
            }

        self.W = tf.get_variable(
            name='W',
            shape=[self.n_hidden + self.embedding_dim, self.n_hidden + self.embedding_dim], ## 600*600
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)

        )
        self.w = tf.get_variable(
            name='w',
            shape=[self.n_hidden + self.embedding_dim, 1], ## 600 *1
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)

        )
        self.Wp = tf.get_variable(
            name='Wp',
            shape=[self.n_hidden, self.n_hidden], ## 300 * 300
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        self.Wx = tf.get_variable(
            name='Wx',
            shape=[self.n_hidden, self.n_hidden], ## 300 * 300
            initializer=tf.random_uniform_initializer(-0.01, 0.01),
            regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )

    def dynamic_rnn(self, cell, inputs, length, max_len, scope_name,model_name):
        ## inputs -> batch_size * max_len * n_hidden
        ## length -> length of each sentence in a batch [of size = batch_size which are total sentences]
        outputs, _ = tf.nn.dynamic_rnn(
            cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        )
        #batch_size = tf.shape(outputs)[0]
        if model_name=="AE": ##taking average of all output vector instead of only taking last vector which is traditional
            outputs = LSTM.reduce_mean(outputs, length)##batch_size * n_hidden  (25 * 300)
            #batch_size = tf.shape(outputs)[0]
            #index = tf.range(0,batch_size)*max_len* (length-1)
            #outputs = tf.gather(tf.reshape(outputs,[-1,self.n_hidden]),index) ##batch_size * n_hidden  (25 * 300)
        return outputs # outputs -> batch_size * max_len * n_hidden (25 * 80 * 300)

    def bi_dynamic_rnn(self, cell, inputs, length, max_len, scope_name,model_name):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell(self.n_hidden),
            cell_bw=cell(self.n_hidden),
            inputs=inputs,
            sequence_length=length,
            dtype=tf.float32,
            scope=scope_name
        )
          
        if model_name=="AE":
            #outputs = tf.concat(outputs, 2)
            #outputs = LSTM.reduce_mean(outputs, length)  # batch_size * 2n_hidden (25 * 600)
            outputs_fw , outputs_bw = outputs
            outputs_bw  = tf.reverse_sequence(outputs_bw,tf.cast(length,tf.int64),seq_dim=1)
            outputs = tf.concat([outputs_fw,outputs_bw],2)
            batch_size = tf.shape(outputs)[0]
            index =  tf.range(0,batch_size)*max_len + (length-1)
            outputs = tf.gather(tf.reshape(outputs,[-1,2*self.n_hidden]),index) #batch_size * 2n_hidden (25*600)
        else:
            outputs = tf.concat(outputs, 2)# batch_size * max_len * 2n_hidden (25*80*600)   
        return outputs

    def AE(self, inputs, target,flag):
        ##inputs :- 25 * 80 * 300    target (aspects):- 25 * 300
        print('Entered AE...')
        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.embedding_dim]) #25 * 1 * 300
        target = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * target # 25  * 80 * 300
        inputs = tf.concat([inputs, target], 2)  #25 * 80 * 600
        inputs = tf.nn.dropout(inputs, keep_prob=1)
        cell = tf.nn.rnn_cell.LSTMCell
        if flag!="bi":
            outputs = self.dynamic_rnn(cell, inputs, self.sen_len, self.max_sentence_len, 'AE','AE')## 25 * 300
        else:
            hiddens = self.bi_dynamic_rnn(cell, inputs, self.sen_len, self.max_sentence_len, 'AE','AE') ##(25 * 600)
            hiddens_fw,hiddens_bw=tf.split(hiddens, num_or_size_splits=2,axis=1)
            #h_t1 =  tf.concat([hiddens_fw, target], 1)
            #h_t2 = tf.concat([hiddens_bw,target],1)
            outputs = tf.math.add(hiddens_fw,hiddens_bw)/2
        return LSTM.softmax_layer(outputs, self.weights['softmax'], self.biases['softmax'])

    def AT(self, inputs, target,flag):
        ## inputs data_matrix and aspect_matrix
        ## inputs :- 25 * 80 * 300
        ## target (aspects) :- 25 * 300
        print('Entered ATAE...')

        batch_size = tf.shape(inputs)[0]
        target = tf.reshape(target, [-1, 1, self.embedding_dim]) ## batch_size * 1 * n_hidden (25 * 1 * 300)
        target = tf.ones([batch_size, self.max_sentence_len, self.embedding_dim], dtype=tf.float32) * target ## for each batch it will create copy (80) of same aspect vectors so that in next step we can concatenate each batch inputs with that many aspect_vectors (25* 80 *300)
        ##concatenation of input word vectors with aspect embeddings
        in_t = tf.concat([inputs, target], 2) ## (25 * 80 * 600)
        in_t = tf.nn.dropout(in_t, keep_prob=1)
        cell = tf.nn.rnn_cell.LSTMCell
        if flag!="bi":
            hiddens = self.dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT','AT') # 25 * 80 * 300
            ##Concatenation of hidden vectors with aspect embeddings
            h_t = tf.reshape(tf.concat([hiddens, target], 2), [-1, self.n_hidden + self.embedding_dim]) #25 * 80 * 600 -> 2000 * 600 [if flag=="!bi"] , else: 25 * 80 * 900 -> 3000 * 600
            #shape_ht = tf.shape(h_t)
        else:
            hidden_vecs = self.bi_dynamic_rnn(cell, in_t, self.sen_len, self.max_sentence_len, 'AT',"AT") # 25*80*600
            hiddens_fw,hiddens_bw=tf.split(hidden_vecs, num_or_size_splits=2, axis=2)
            hiddens = tf.math.add(hiddens_fw,hiddens_bw)/2
            h_avg = tf.concat([hiddens,target],2)
            h_t = tf.reshape(h_avg,[-1,self.n_hidden+self.embedding_dim])
        ##### for flag=="!bi" :- dimensionas are shown below ####
        M = tf.tanh(tf.matmul(h_t, self.W)) ##(2000*600), #W:- (600 * 600), w:- (600 * 1)
        Mdotw = tf.reshape(tf.matmul(M,self.w), [-1, 1, self.max_sentence_len]) ##2000*1 -> 25 *1*80
        alpha = LSTM.softmax(Mdotw, self.sen_len, self.max_sentence_len)  #25 * 1 * 80
        self.alpha = tf.reshape(alpha, [-1, self.max_sentence_len]) #25 * 80
        r = tf.reshape(tf.matmul(alpha, hiddens), [-1, self.n_hidden])  #25 * 300
        index = tf.range(0, batch_size) * self.max_sentence_len + (self.sen_len - 1)
        ## hn is last hidden vector of sentence
        hn = tf.gather(tf.reshape(hiddens, [-1, self.n_hidden]), index)  # batch_size * n_hidden (25*300)
        h_star = tf.tanh(tf.matmul(r, self.Wp) + tf.matmul(hn, self.Wx))#Wp:- 300*300, Wx:- 300*300, h_star:- 25*300
        return LSTM.softmax_layer(h_star, self.weights['softmax'], self.biases['softmax'])

    @staticmethod
    def softmax_layer(inputs, weights, biases):
        ##inputs :- 25 * 300
        ##weights (Ws) :- 300 * 3
        ##biases (bs) :- list of length 3 or 3*1
        with tf.name_scope('softmax'):
            outputs = tf.nn.dropout(inputs, keep_prob=1)
            predict = tf.matmul(outputs, weights) + biases
            predict = tf.nn.softmax(predict)
        return predict

    @staticmethod
    def reduce_mean(inputs, length):
        ##Word vectors of each sentence will replaced by their mean
        ##return :- (25*300)
        ##inputs :- #25 * 80 * 300
        ##length :- list containing len of all sent in a batch [list of length 25]
        length = tf.cast(tf.reshape(length, [-1, 1]), tf.float32) + 1e-9
        inputs = tf.reduce_sum(inputs, 1, keepdims=False) / length ##(25*300)
        return inputs

    @staticmethod
    def softmax(inputs, length, max_length):
        ##inputs :- 25 * 1 * 80
        ##length :- list of length 25(list contain all sentence len)
        ##max_length :- max sentence len (80)
        inputs = tf.cast(inputs, tf.float32)
        max_axis = tf.reduce_max(inputs, 2, keepdims=True)## 25 * 1 * 1
        inputs = tf.exp(inputs - max_axis)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keepdims=True) + 1e-9
        return inputs / _sum

    def run(self):
        inputs = tf.nn.embedding_lookup(self.word_embedding, self.x) ##25 * 80 * 300
        aspect = tf.nn.embedding_lookup(self.aspect_embedding, self.aspect_id) ## 25 * 300
        #print(self.model_name)
        if self.model_name=="AE":
            prob = self.AE(inputs,aspect,self.model_type)
        else:
            prob = self.AT(inputs,aspect,self.model_type)

        with tf.name_scope('loss'):
            reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            cost = - tf.reduce_mean(tf.cast(self.y, tf.float32) * tf.log(prob)) + sum(reg_loss)

        with tf.name_scope('train'):
            global_step = tf.Variable(0, name="tr_global_step", trainable=False)
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=global_step)

        with tf.name_scope('predict'):
            true_y = tf.argmax(self.y, 1)
            pred_y = tf.argmax(prob, 1)
            correct_pred = tf.equal(pred_y,true_y)
            accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
            _acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        with tf.Session() as sess:
            title = 'model_name:{}-model_type:{}-b:{}-r:{}-l2:{}-sen:{}-dim:{}-h:{}-c:{}'.format(
                self.model_name,
                self.model_type,
                FLAGS.batch_size,
                FLAGS.learning_rate,
                FLAGS.l2_reg,
                FLAGS.max_sentence_len,
                FLAGS.embedding_dim,
                FLAGS.n_hidden,
                FLAGS.n_class
            )

            init = tf.global_variables_initializer()
            sess.run(init)

            tr_x, tr_sen_len, tr_target_word, tr_y = load_inputs_at(
                FLAGS.train_file_path,
                self.word_id_mapping,
                self.aspect_id_mapping,
                self.max_sentence_len
            )
            te_x, te_sen_len, te_target_word, te_y = load_inputs_at(
                FLAGS.test_file_path,
                self.word_id_mapping,
                self.aspect_id_mapping,
                self.max_sentence_len
            )

            max_acc = 0.
            max_alpha = None
            max_ty, max_py = None, None
            for i in range(self.n_iter):
                for train,_ in self.get_batch_data(tr_x, tr_sen_len, tr_y, tr_target_word, self.batch_size):
                    #print(num)
                    #_shape = sess.run(shape_ht,feed_dict=train)
                    #print(_shape)
                    _, step= sess.run([optimizer, global_step], feed_dict=train)
                    
                acc, loss, cnt = 0., 0., 0
                flag = True
                alpha = None
                ty, py = None, None
                ## for each batch do the following
                ## test is test data matrix and num is number of samples/instances in test data
                for test, num in self.get_batch_data(te_x, te_sen_len, te_y, te_target_word, 2000, False):
                    if self.model_name =="AE":
                        _loss , _acc ,_ ,ty,py = sess.run([cost,accuracy,global_step,true_y,pred_y],feed_dict=test)
                    else:
                        _loss, _acc, _, alpha, ty, py = sess.run([cost, accuracy, global_step, self.alpha, true_y, pred_y], feed_dict=test)
                    acc += _acc
                    loss += _loss * num
                    cnt += num
                    if flag:
                        flag = False
                        alpha = alpha
                        #print(alpha)
                        ty = ty
                        py = py

                print('all samples={}, correct prediction={}'.format(cnt, acc))
                ##per iteration averge test acc and average mini-batch loss
                print('Iter {}: mini-batch loss={:.6f}, test acc={:.6f}'.format(i, loss / cnt, acc / cnt))
                if acc / cnt > max_acc:
                    max_acc = acc / cnt
                    max_alpha = alpha
                    #print(max_alpha)
                    max_ty = ty
                    max_py = py

            print('Optimization Finished! Max acc={}'.format(max_acc))
            p,r,f,_ = score(max_ty,max_py,average=None,labels=[1,2,0])
            print('P_pos={:.6f}, P_neg={:.6f}, P_nut={:.6f}'.format(p[0],p[1],p[2]))
            print('R_pos={:.6f}, R_neg={:.6f}, R_nut={:.6f}'.format(r[0],r[1],r[2]))
            print('F_pos={:.6f}, F_neg={:.6f}, F_nut={:.6f}'.format(f[0],f[1],f[2]))

            out_path = "results/"
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            fd = open(out_path+'metric_'+title+".txt",'w')

            fd.write('Optimization Finished! Max acc={}'.format(max_acc)+"\n"
                     +'P_pos={:.6f}, P_neg={:.6f}, P_nut={:.6f}'.format(p[0],p[1],p[2])+"\n"
                     +'R_pos={:.6f}, R_neg={:.6f}, R_nut={:.6f}'.format(r[0],r[1],r[2])+"\n"
                     +'F_pos={:.6f}, F_neg={:.6f}, F_nut={:.6f}'.format(f[0],f[1],f[2]))
            fd.close()
            if self.model_name!="AE":
                fd = open(out_path+'alpha_'+title+'.txt','w')
                fd.write(str(max_alpha))
                fd.close()
            print('Learning_rate={}, iter_num={}, batch_size={}, hidden_num={}, l2={}'.format(
                self.learning_rate,
                self.n_iter,
                self.batch_size,
                self.n_hidden,
                self.l2_reg
            ))

    def get_batch_data(self, x, sen_len, y, target_words, batch_size, is_shuffle=True):
        ## x, data matrix which contain each sentence as of length 80 [padding included] and sentence vector contains ids and not words
        ## sen_len , list of lengths of all sentences
        ## y , label matrix which contain one-hot vector corresponding to each label
        ## target_words , list of ids of all aspect terms
        for index in batch_index(len(y), batch_size, 1, is_shuffle):
            #print("indexes: ",index)
            feed_dict = {
                self.x: x[index],
                self.y: y[index],
                self.sen_len: sen_len[index],
                self.aspect_id: target_words[index]
            }
            yield feed_dict , len(index) ##returning batch data and batch size

data="data"
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('batch_size', 25, 'number of example per batch')
tf.app.flags.DEFINE_integer('n_hidden', 300, 'number of hidden unit')
tf.app.flags.DEFINE_float('learning_rate', 0.05,'learning rate') ##0.001 for AE with avgeraging and 0.005 for everything else
tf.app.flags.DEFINE_integer('n_class', 3, 'number of distinct class')
tf.app.flags.DEFINE_integer('max_sentence_len', 80, 'max number of tokens per sentence')
tf.app.flags.DEFINE_float('l2_reg', 0.001, 'l2 regularization')
tf.app.flags.DEFINE_integer('n_iter',25, 'number of train iter')

tf.app.flags.DEFINE_string('train_file_path', data+'/restaurant/rest_2014_train.txt', 'training file')
tf.app.flags.DEFINE_string('test_file_path', data+'/restaurant/rest_2014_test.txt', 'testing file')
tf.app.flags.DEFINE_string('embedding_file_path', data+'/restaurant/rest_2014_word_embedding_300.txt', 'embedding file')
tf.app.flags.DEFINE_string('word_id_file_path', data+'/restaurant/word_id.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('aspect_id_file_path', data+'/restaurant/aspect_id.txt', 'word-id mapping file')
tf.app.flags.DEFINE_string('model_type2',"bi","bidirectional LSTM")
tf.app.flags.DEFINE_string('model_name2',"AT","ATAE LSTM")
tf.app.flags.DEFINE_string('model_type1',"!bi","standard LSTM")
tf.app.flags.DEFINE_string('model_name1',"AE","AE LSTM")
def main(_):
    lstm = LSTM(
        embedding_dim=FLAGS.embedding_dim,
        batch_size=FLAGS.batch_size,
        n_hidden=FLAGS.n_hidden,
        learning_rate=FLAGS.learning_rate,
        n_class=FLAGS.n_class,
        max_sentence_len=FLAGS.max_sentence_len,
        l2_reg=FLAGS.l2_reg,
        n_iter=FLAGS.n_iter,
        model_type = FLAGS.model_type2,
        model_name = FLAGS.model_name2
    )
    lstm.run()


if __name__ == '__main__':
    tf.app.run()

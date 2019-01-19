from data import *
from perceptron import *
from features import *
import pickle

MODEL1_WEIGHTS = 'cache/basic_N1.pickle'
MODEL2_WEIGHTS = 'cache/complex_N1.pickle'


def predict(data, w, perceptron):
    """predict function"""
    pred_list = []
    for idx, sentence in enumerate(data.sentences):
        pred_list.append(perceptron.sentence_inference(w, sentence.sentence_len, perceptron._f_dict_list[idx]))
    return pred_list


# model1 -> basic features
# model2 -> complex features

# extract features from training file
train_data = Data('train.labeled', is_labeled=True)
train_features_model1 = BasicFeatures(train_data.vocab_list, train_data.pos_list, train_data.word_pos_pairs)
train_features_model2 = ComplexFeatures(train_data.vocab_list, train_data.pos_list, train_data.word_pos_pairs)

# extract features from competition file
comp_data = Data('comp.unlabeled', is_labeled=False)
comp_m1_perceptron = Perceptron(comp_data, train_features_model1)
comp_m2_perceptron = Perceptron(comp_data, train_features_model2)

# load models weights
model1_w = pickle.load(open(MODEL1_WEIGHTS, 'rb'))
model2_w = pickle.load(open(MODEL2_WEIGHTS, 'rb'))

# predict
model1_pred = predict(comp_data, model1_w, comp_m1_perceptron)
model2_pred = predict(comp_data, model2_w, comp_m2_perceptron)

# create output files
m1_fh = open('../comp_m1_305219768.wtag', 'w')
m2_fh = open('../comp_m2_305219768.wtag', 'w')

# write predictions to output files
with open('comp.unlabeled') as comp_fh:
    line = comp_fh.readline()
    sen_num = 0
    word_num = 0
    while line:
        word_num += 1
        args = line.split()
        if len(args) > 1:
            for idx in range(len(args) - 1):
                if idx == 6:
                    m1_fh.write(str(model1_pred[sen_num][word_num]) + '\t')
                    m2_fh.write(str(model2_pred[sen_num][word_num]) + '\t')
                else:
                    m1_fh.write(args[idx] + '\t')
                    m2_fh.write(args[idx] + '\t')
            m1_fh.write(args[-1] + '\r\n')
            m2_fh.write(args[-1] + '\r\n')
        else:
            sen_num += 1
            word_num = 0
            m1_fh.write('\r\n')
            m2_fh.write('\r\n')
        line = comp_fh.readline()

# close output files
m1_fh.close()
m2_fh.close()

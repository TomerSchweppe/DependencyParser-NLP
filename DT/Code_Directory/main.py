# !/usr/bin/env python
from data import *
from perceptron import *
from features import *
import argparse
import pickle
import time


def evaluate(labeled_data, w, perceptron):
    """evaluate model accuracy per word"""
    total = 0
    correct = 0
    for idx, sentence in enumerate(labeled_data.sentences):
        ground_truth = sentence.dependency_tree()
        predicted = perceptron.sentence_inference(w, sentence.sentence_len, perceptron._f_dict_list[idx])
        for x in range(1, sentence.sentence_len):
            total += 1
            if predicted[x] == ground_truth[x]:
                correct += 1
    return correct / total


if __name__ == '__main__':
    """main program"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help="load trained weights")
    parser.add_argument("--N", help="number of iterations", default=1)
    parser.add_argument("--features", help="features type basic/complex", default='basic')
    parser.add_argument("--train_data", help="path to training data", default='train.labeled')
    args = parser.parse_args()

    N = int(args.N)
    features_type = args.features

    # init train
    train_data = Data(args.train_data, is_labeled=True)
    if (args.weights and 'basic' in args.weights) or (not args.weights and features_type == 'basic'):
        train_features = BasicFeatures(train_data.vocab_list, train_data.pos_list, train_data.word_pos_pairs)
    else:
        train_features = ComplexFeatures(train_data.vocab_list, train_data.pos_list, train_data.word_pos_pairs)

    if args.weights:  # load trained weights
        train_w = pickle.load(open(args.weights, 'rb'))
    else:
        # init train
        start = time.time()
        print('extract train features')
        train_perceptron = Perceptron(train_data, train_features)
        print('extract ended', time.time() - start)

        # learn train weights
        start = time.time()
        print('learn model weights')
        train_w = train_perceptron.train(N)
        print('learning ended: ', time.time() - start)
        # train evaluation
        start = time.time()
        print('train evaluation')
        train_accuracy = evaluate(train_data, train_w, train_perceptron)
        print('train accuracy: ', train_accuracy)
        print('evaluation ended: ', time.time() - start)
        # save train weights
        pickle.dump(train_w, open('cache/' + features_type + '_N' + str(N) + '.pickle', 'wb'))

    # init test
    start = time.time()
    print('extract test features')
    test_data = Data('test.labeled', is_labeled=True)
    test_perceptron = Perceptron(test_data, train_features)
    print('extract ended', time.time() - start)

    start = time.time()
    print('test evaluation')
    test_accuracy = evaluate(test_data, train_w, test_perceptron)
    print('test accuracy: ', test_accuracy)
    print('evaluation ended: ', time.time() - start)

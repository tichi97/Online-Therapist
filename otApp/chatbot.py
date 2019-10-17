import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()  # gets the root of the word

import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle


def build_chatbox():
    with open("intents.json") as file:
    data = json.load(file)


try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

    except:
        words = []
        labels = []
        docs_x = []  # all the patterns
        docs_y = []  # tage for the words

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
        words = sorted(list(set(words)))  # remove duplicates

        labels = sorted(labels)

        # one-hot encoded
        training = []
        output = []

        out_empty = [0 for _ in range(len(labels))]

        for x, doc in enumerate(docs_x):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]

            for w in words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1
            training.append(bag)
            output.append(output_row)

        # change to array to feed into tensorflow
        training = numpy.array(training)
        output = numpy.array(output)

        with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)

    tf.reset_default_graph()

    # neural network
    net = tflearn.input_data(shape=[None, len(training[0])])
    # hidden layers to figure out whats being talked about
    net = tflearn.fully_connected(net, 8)  # 8 neurons
    net = tflearn.fully_connected(net, 8)
    # softmax -> gives higher probablity to the label/tag that the model thinks the input is
    net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    # train model
    model = tflearn.DNN(net)

    try:
        model.load("model.tflearn")
    except:
        #                         n_epoch:the number of times the model will see the training data
        model.fit(training, output, n_epoch=1000,
                  batch_size=8, show_metric=True)
        model.save("model.tflearn")

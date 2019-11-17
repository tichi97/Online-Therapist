import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()  # gets the root of the word

from flask import render_template, url_for
import numpy
import tflearn
import tensorflow as tf
import random
import json
import pickle
from otApp.emotion import check_emo

with open("otApp/intents.json") as file:
    data = json.load(file)

try:
    with open("otApp/data.pickle", "rb") as f:
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

# ------------------------------------------------------------------------------------------
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
    # n_epoch:the number of times the model will see the training data
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")
# ----------------------------------------------------------


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)


flag = 0


def chatbot(inp):
    print("Start talking with the bot! (type quit to stop)")
    # if flag == 1:
    #     inp_pred = check_emo(inp)
    #     return inp_pred
    while True:
        # inp = input("You: ")
        if inp.lower() == "quit":
            break
        # a bunch of probabilities
        result = model.predict([bag_of_words(inp, words)])[0]
        # returns index of the greatest probability
        result_index = numpy.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                # if tag == "feelings":
                #     url_for("feelings")
                # else:
                #     # print(random.choice(responses))
                    return random.choice(responses), tag
        else:
            # print("I'm sorry, I don't understand")
            return "I'm sorry, I don't understand. Let's talk about how you are feeling.", tag


# print(chat())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from tensorflow.python.keras.preprocessing import sequence
from nltk.translate.bleu_score import *
from util import *
from random import randint

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('phase', 'train', "Which operation to run.")
logger = logging.getLogger(__name__)
# Comment out the below statement to avoid printing info logs to console
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


model_path = './models'
feature_path = './data/feats.npy'
captions_path = './data/results.token'
vgg_path = './data/vgg16.tfmodel'


def get_data(captions_path, feature_path):
    captions = pd.read_table(captions_path, sep='\t', header=None, names=['image', 'caption'])
    return np.load(feature_path,'r'), captions['caption'].values


def build_vocab(sentences, threshold=30):

    logger.info("Building vocabulary")
    word_counts = {}
    nsents = 0
    for sentence in sentences:
      nsents += 1
      for word in sentence.lower().split(' '):
        word_counts[word] = word_counts.get(word, 0) + 1
    vocab = [word for word in word_counts if word_counts[word] >= threshold]
    logger.info("Vocab size: %i", len(vocab))

    idx_to_word = {}
    idx_to_word[0] = '.'  
    word_to_idx = {}
    word_to_idx['<START>'] = 0 
    idx = 1
    for word in vocab:
      word_to_idx[word] = idx
      idx_to_word[idx] = word
      idx += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0 * word_counts[idx_to_word[i]] for i in idx_to_word])
    bias_init_vector /= np.sum(bias_init_vector) 
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) 
    return word_to_idx, idx_to_word, bias_init_vector.astype(np.float32)


class Caption_Generator():

    def __init__(self, input_dims, embed_dims, hidden_dims, batch_size, n_lstm_steps, n_words, init_bias=None):

        self.input_dims = input_dims
        self.embed_dims = embed_dims
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words

        with tf.device("/gpu:0"):
            self.word_embedding = tf.Variable(tf.random_uniform(
                    [self.n_words, self.embed_dims], -0.1, 0.1), name='word_embedding')
        self.word_embedding_bias = tf.Variable(tf.zeros([embed_dims]), name='word_embedding_bias')
        self.lstm = tf.contrib.rnn.BasicLSTMCell(hidden_dims)
        self.img_embedding = tf.Variable(tf.random_uniform(
                [input_dims, hidden_dims], -0.1, 0.1), name='img_embedding')
        self.img_embedding_bias = tf.Variable(tf.zeros([hidden_dims]), name='img_embedding_bias')
        self.word_encoding = tf.Variable(tf.random_uniform([hidden_dims, n_words], -0.1, 0.1), name='word_encoding')
        if init_bias is not None:
            self.word_encoding_bias = tf.Variable(init_bias, name='word_encoding_bias')
        else:
            self.word_encoding_bias = tf.Variable(tf.zeros([n_words]), name='word_encoding_bias')

    def build_model(self):

        image_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.input_dims])
        caption_placeholder = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])
        image_embedding = tf.matmul(image_placeholder, self.img_embedding) + self.img_embedding_bias
        # initial state of LSTM
        state = self.lstm.zero_state(self.batch_size, dtype=tf.float32)

        total_loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps): 
                if i > 0:
                    with tf.device("/gpu:0"):
                        current_embedding = tf.nn.embedding_lookup(self.word_embedding, caption_placeholder[:,i-1]) + self.word_embedding_bias
                else: 
                    current_embedding = image_embedding
                if i > 0: 
                    tf.get_variable_scope().reuse_variables()
                output, state = self.lstm(current_embedding, state)
                
                if i > 0:
                    labels = tf.expand_dims(caption_placeholder[:, i], 1)
                    idx_range=tf.range(0, self.batch_size, 1)
                    idxs = tf.expand_dims(idx_range, 1)
                    concat = tf.concat([idxs, labels],1)
                    onehot = tf.sparse_to_dense(
                            concat, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

                    logit = tf.matmul(output, self.word_encoding) + self.word_encoding_bias
                    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit, labels=onehot)
                    xentropy = xentropy * mask[:,i]
                    loss = tf.reduce_sum(xentropy)
                    total_loss += loss

            total_loss = total_loss / tf.reduce_sum(mask[:,1:])
            return total_loss, image_placeholder, caption_placeholder, mask

    def build_generator(self, maxlen, batch_size=1):

        image_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.input_dims])
        image_embedding = tf.matmul(image_placeholder, self.img_embedding) + self.img_embedding_bias
        state = self.lstm.zero_state(batch_size,dtype=tf.float32)

        all_words = []
        with tf.variable_scope("RNN"):
            output, state = self.lstm(image_embedding, state)
            previous_word = tf.nn.embedding_lookup(self.word_embedding, [0]) + self.word_embedding_bias
            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()
                out, state = self.lstm(previous_word, state)
                logit = tf.matmul(out, self.word_encoding) + self.word_encoding_bias
                best_word = tf.argmax(logit, 1)
                with tf.device("/gpu:0"):
                    previous_word = tf.nn.embedding_lookup(self.word_embedding, best_word)
                previous_word += self.word_embedding_bias
                all_words.append(best_word)

        return image_placeholder, all_words


def train(learning_rate=0.001, continue_training=True, n_epochs=20, batch_size=128):

    #tf.reset_default_graph()
    feats, captions = get_data(captions_path, feature_path)
    logger.info("Number of examples: %i", captions.shape[0])
    logger.info("Image embedding dimensions: %i", feats.shape[1])
    word_to_idx, idx_to_word, init_bias = build_vocab(captions)
    np.save('data/idx_to_word', idx_to_word)

    index = (np.arange(len(feats)).astype(int))
    np.random.shuffle(index)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    n_words = len(word_to_idx)
    maxlen = np.max( [x for x in map(lambda x: len(x.split(' ')), captions) ] )
    caption_generator = Caption_Generator(input_dims=4096, hidden_dims=256, embed_dims=256,
                                          batch_size=batch_size, n_lstm_steps=maxlen+2, n_words=n_words,
                                          init_bias=init_bias)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=100)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                       int(len(index)/batch_size), 0.95)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.global_variables_initializer().run()

    if continue_training:
        saver.restore(sess,tf.train.latest_checkpoint(model_path))

    for epoch in range(n_epochs):
        for start, end in zip( range(0, len(index), batch_size), range(batch_size,
                              len(index), batch_size)):
            current_feats = feats[index[start:end]]
            current_captions = captions[index[start:end]]
            current_caption_ind = [x for x in map(lambda cap:
                [word_to_idx[word] for word in cap.lower().split(' ')[:-1] if word in word_to_idx], current_captions)]
            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] )
            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array([x for x in map(lambda x: (x != 0).sum()+2, current_caption_matrix )])

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats.astype(np.float32),
                sentence : current_caption_matrix.astype(np.int32),
                mask : current_mask_matrix.astype(np.float32)
                })

            logger.info("Current cost: %.4f, Epoch: %i, Iter: %i", loss_value, epoch, start)
        logger.info("Saving model from epoch %i", epoch)
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)


def crop_image(x, target_height=227, target_width=227, as_float=True):

    image = cv2.imread(x)
    if as_float:
        image = cv2.imread(x).astype(np.float32)

    if len(image.shape) == 2:
        image = np.tile(image[:,:,None], 3)
    elif len(image.shape) == 4:
        image = image[:,:,:,0]

    height, width, rgb = image.shape
    if width == height:
        resized_image = cv2.resize(image, (target_height,target_width))

    elif height < width:
        resized_image = cv2.resize(image, (int(width * float(target_height)/height), target_width))
        cropping_length = int((resized_image.shape[1] - target_height) / 2)
        resized_image = resized_image[:,cropping_length:resized_image.shape[1] - cropping_length]

    else:
        resized_image = cv2.resize(image, (target_height, int(height * float(target_width) / width)))
        cropping_length = int((resized_image.shape[0] - target_width) / 2)
        resized_image = resized_image[cropping_length:resized_image.shape[0] - cropping_length,:]

    return cv2.resize(resized_image, (target_height, target_width))

def read_image(path):

     img = crop_image(path, target_height=224, target_width=224)
     if img.shape[2] == 4:
         img = img[:,:,:3]
     img = img[None, ...]
     return img


def bleu_score(sentence):
    
    sentence = sentence.split()
    index = randint(0, 158914)
    reference1, reference2, reference3, reference4, reference5 = get_image_caption(index)
    reference1 = reference1.split()
    reference2 = reference2.split()
    reference3 = reference3.split()
    reference4 = reference4.split()
    reference5 = reference5.split()

    chencherry = SmoothingFunction() # SmoothingFunction object.smoothing techniques for segment-level BLEU scores.
    # Use method7.
    BLEU_1 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                sentence, weights=[0.25], smoothing_function=chencherry.method7) # list(str).
    print("BLEU-1: %f"%(BLEU_1))

    BLEU_2 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
            sentence, weights=(0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
    print("BLEU-2: %f"%(BLEU_2))

    BLEU_3 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                sentence, weights=(0.25, 0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
    print("BLEU-3: %f"%(BLEU_3))

    BLEU_4 = sentence_bleu([reference1, reference2, reference3, reference4, reference5],
                sentence, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method7) # list(str).
    print("BLEU-4: %f"%(BLEU_4))
    return


def generate_image_caption(image_path='./test_images/test1.jpg', train_model=True):

    if train_model is True:
        try:
            logger.info("Training image caption generator")
            train()
        except KeyboardInterrupt:
            logger.info("Exiting training")
    tf.reset_default_graph()
    with open(vgg_path,'rb') as f:
        fileContent = f.read()
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fileContent)

    images = tf.placeholder("float32", [1, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={"images":images})
    idx_to_word = np.load('data/idx_to_word.npy').tolist()
    n_words = len(idx_to_word)
    maxlen = 15
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(graph=graph, config=config)
    caption_generator = Caption_Generator(input_dims=4096, hidden_dims=256, embed_dims=256,
                                          batch_size=1, n_lstm_steps=maxlen+2, n_words=n_words)
    graph = tf.get_default_graph()
    image, generated_words = caption_generator.build_generator(maxlen=maxlen)

    feat = read_image(image_path)
    fc7 = sess.run(graph.get_tensor_by_name("import/Relu_1:0"), feed_dict={images:feat})
    saver = tf.train.Saver()
    sanity_check = False
    if not sanity_check:
        saved_path = tf.train.latest_checkpoint(model_path)
        saver.restore(sess, saved_path)
    else:
        tf.global_variables_initializer().run()
    generated_word_index= sess.run(generated_words, feed_dict={image:fc7})
    generated_word_index = np.hstack(generated_word_index)
    generated_words = [idx_to_word[x] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1
    generated_words = generated_words[:punctuation]
    generated_sentence = ' '.join(generated_words)
    bleu_score(generated_sentence)
    return generated_sentence

if __name__=="__main__":
    if FLAGS.phase == 'train':
        print(generate_image_caption())
    elif FLAGS.phase == 'test':
        print(generate_image_caption(image_path='./test_images/test.jpg', train_model=False))

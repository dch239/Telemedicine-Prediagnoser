import numpy as np
import pickle 
import random
import json
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import progressbar
import urllib.request
import zipfile

from sklearn import preprocessing as pe
from keras.models import *
from keras.layers import *
import keras

from keras import Model
from keras.layers import Layer
import keras.backend as K
from keras.layers import Input, Dense, SimpleRNN,LSTM
from keras.models import Sequential


class ChatBot():
  def __init__(self) -> None:
    self.docs_x = []
    self.docs_y = []
    self.labels= []
    self.y = None
    self.diseases = 0
    self.word_count = 0
    self.max_symptoms_count = 0
    self.embeddings_index = dict()
    self.rt = RegexpTokenizer(r'[^\W_]+|[^\W_\s]+')
    self.stop = set(stopwords.words('english') + list(string.punctuation))
    self.label_encoder = pe.LabelEncoder()
    self.model = None
    self.batch_size = 8
    self.model_history = None
    self.pbar = None
    self.embeddings_input_matrix = []
    

  #Loading Data
  def load_data(self):
    with open("intents.json") as file:
      data = json.load(file)
    

    #Looping through our data
    for intent in data['intents']:
      words = []
      for pattern in intent['patterns']:
        pattern = pattern.lower()
        #print(pattern)
        #Creating a list of words
        ###wrds = rt.tokenize(pattern)
        wrds = [i for i in self.rt.tokenize(pattern) if i not in self.stop]
        words.append(wrds)
      self.docs_x.append(words)
      self.docs_y.append(intent['tag'])

  def print_docs(self):
    print(self.docs_x)
    print(self.docs_y)
    print(len(self.docs_y))

  def nltk_download():
    nltk.download('punkt')
    nltk.download('stopwords')

  def get_glove_embeddings(self):
    try: 
      f = open('./glove.6B.300d.txt', encoding="utf8")
      for line in f:
          word, coefs = line.split(maxsplit=1)
          coefs = np.fromstring(coefs, "f", sep=" ")
          self.embeddings_index[word] = coefs

      print("Found %s word vectors." % len(self.embeddings_index))
    
    except:
      print("Downloading the Embeddings")
      #pbar = None
      def show_progress(block_num, block_size, total_size):
          if self.pbar is None:
              self.pbar = progressbar.ProgressBar(maxval=total_size)
              self.pbar.start()

          downloaded = block_num * block_size
          if downloaded < total_size:
              self.pbar.update(downloaded)
          else:
              self.pbar.finish()
              pbar = None

          #urllib.request.urlretrieve(model_url, model_file, show_progress)
      urllib.request.urlretrieve('https://nlp.stanford.edu/data/glove.6B.zip','glove.6B.zip', show_progress)
      with zipfile.ZipFile("/content/glove.6B.zip", 'r') as zip_ref:
        zip_ref.extractall("/content/")
      self.get_glove_embeddings()


      
  def test_embeddings(self):
    #testing embeddings
    embedding_vector = self.embeddings_index.get("apple")
    print(embedding_vector.shape)
    arr = np.zeros(300)

  def get_counts(self): 
    for symptoms_of_1_disease in self.docs_x:
      self.diseases += 1
      self.max_symptoms_count = max(self.max_symptoms_count, len(symptoms_of_1_disease))
      for individual_symptom in symptoms_of_1_disease:
        for word in individual_symptom:
          self.word_count+=1
    print(self.diseases)
    print(self.max_symptoms_count)
    print(self.word_count)

  def make_embeddings_matrix(self):
    hits=0
    misses=0
    i = 1
    for symptoms_of_1_disease in self.docs_x:
      symptoms_embedding = np.zeros((self.max_symptoms_count-len(symptoms_of_1_disease), 300))
      for individual_symptom in symptoms_of_1_disease:
        individual_symptom_embedding = np.zeros(300)
        for word in individual_symptom:
          word_embedding = self.embeddings_index.get(word)
          if word_embedding is not None:
            hits += 1
            individual_symptom_embedding += word_embedding
          else:
            misses += 1
        symptoms_embedding = np.append(symptoms_embedding, [individual_symptom_embedding], axis = 0)
      self.embeddings_input_matrix.append(symptoms_embedding)

    self.embeddings_input_matrix = np.array(self.embeddings_input_matrix)      
    print(f"hits = {hits}, misses = {misses}, ratio = {hits/(hits+misses)}")
    print(self.embeddings_input_matrix.shape)

  #label encoding y axis
  def encode_y(self):  
    self.y =  self.label_encoder.fit_transform(self.docs_y)
    self.y = np.asarray(self.y)
    print(self.y)

  def define_model(self):
    self.model = self.create_LSTM_with_attention(hidden_units=256, dense_units=41, input_shape=(self.max_symptoms_count,300), activation='softmax')
    self.model.summary()

  def train(self):
    self.model_history = self.model.fit([self.embeddings_input_matrix], self.y, epochs=20, batch_size=8,verbose=1)

  def train_with_datagen(self):
    train_gen = DataGen(
        self.docs_x,
        self.y,
        self.batch_size,
        self.embeddings_index,
        self.max_symptoms_count
        )
    train_steps = self.embeddings_input_matrix.shape[0]//self.batch_size

    self.model_history = self.model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=200,
        verbose=1)

  def predict_from_list(self, symptoms_of_1_disease):
    hits = 0
    misses = 0
    symptoms_embedding = np.zeros((self.max_symptoms_count-len(symptoms_of_1_disease), 300))
    for individual_symptom in symptoms_of_1_disease:
      individual_symptom_embedding = np.zeros(300)
      words = [i for i in self.rt.tokenize(individual_symptom) if i not in self.stop]
      print(words)
      for word in words:
        word_embedding = self.embeddings_index.get(word)
        if word_embedding is not None:
          hits += 1
          individual_symptom_embedding += word_embedding
        else:
          misses += 1
      symptoms_embedding = np.append(symptoms_embedding, [individual_symptom_embedding], axis = 0)
    #print(symptoms_embedding.shape)
    print(f"Of total words in symptoms: hits = {hits}, misses = {misses}, ratio = {hits/(hits+misses)}")
    test_data = np.array([symptoms_embedding])
    print(test_data.shape)
    pred = self.model.predict(test_data)
    pos = np.argmax(pred, axis=1)
    return self.label_encoder.inverse_transform([pos[0]])[0]

  def predict_from_sentence(self,symptoms):
    print(symptoms)
    hits = 0
    misses = 0
    symptoms_embedding = np.zeros((self.max_symptoms_count, 300))
    words = [i for i in self.rt.tokenize(symptoms) if i not in self.stop]
    print(words)
    for word in words:
      if hits == 17:
        break
      word_embedding = self.embeddings_index.get(word)
      if word_embedding is not None:
        hits += 1
        symptoms_embedding[self.max_symptoms_count-hits] += word_embedding
      else:
        word_embedding = np.zeros(300)
        misses += 1
      
    print(symptoms_embedding.shape)
    print(f"Of total words in symptoms: hits = {hits}, misses = {misses}, ratio = {hits/(hits+misses)}")
    test_data = np.array([symptoms_embedding])
    print(test_data.shape)
    pred = self.model.predict(test_data)
    pos = np.argmax(pred, axis=1)
    return self.label_encoder.inverse_transform([pos[0]])[0]

  def create_RNN_with_attention(hidden_units, dense_units, input_shape, activation):
      x=Input(shape=input_shape)
      RNN_layer = SimpleRNN(hidden_units, return_sequences=True, activation=activation)(x)
      attention_layer = attention()(RNN_layer)
      outputs=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
      model=Model(x,outputs)
      model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])    
      return model 
         
  def create_LSTM_with_attention(self, hidden_units, dense_units, input_shape, activation):
      x=Input(shape=input_shape)
      LSTM_layer = LSTM(hidden_units, return_sequences=True, activation='relu')(x)
      attention_layer = attention()(LSTM_layer)
      outputs=Dense(dense_units, trainable=True, activation=activation)(attention_layer)
      model=Model(x,outputs)
      model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])    
      return model

  

class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                              initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                              initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)

    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context

class DataGen(keras.utils.all_utils.Sequence):
    def __init__(self, docs_x, output, batch_size, embeddings_index, max_symptoms_count):
        self.docs_x = docs_x
        self.output = output
        self.batch_size = batch_size
        self.on_epoch_end()
        self.hits=0
        self.misses=0
        self.embeddings_index = embeddings_index
        self.max_symptoms_count = max_symptoms_count
    
    def __getitem__(self, index):
        if((index+1)*self.batch_size > len(self.docs_x)):
            self.batch_size = len(self.docs_x) - index*self.batch_size
        
        inp_batch = self.docs_x[index*self.batch_size : (index+1)*self.batch_size]
        #print()
        inp_batch_embeddings = [] #np.zeros(shape = (self.batch_size, max_symptoms_count, 300))
        for symptoms_of_1_disease in inp_batch:
          random.shuffle(symptoms_of_1_disease)
          symptoms_embedding = np.zeros((len(symptoms_of_1_disease), 300))
          j = 0
          for individual_symptom in symptoms_of_1_disease:
            individual_symptom_embedding = np.zeros(300)
            for word in individual_symptom:
              word_embedding = self.embeddings_index.get(word)
              if word_embedding is not None:
                self.hits += 1
                individual_symptom_embedding += word_embedding
              else:
                self.misses += 1
            symptoms_embedding[j] = individual_symptom_embedding #new code add padding using DataGen later
            j = j + 1
          symptoms_embedding = np.pad(symptoms_embedding, [((self.max_symptoms_count - symptoms_embedding.shape[0]), 0), (0, 0)], mode='constant', constant_values=0)
          inp_batch_embeddings.append(symptoms_embedding)
        
        inp_batch_embeddings = np.array(inp_batch_embeddings)

        out_batch = self.output[index*self.batch_size : (index+1)*self.batch_size]
     
        return inp_batch_embeddings, out_batch
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(len(self.docs_x)/float(self.batch_size)))



def train_chatbot_first_time():
    print("Hello. Let's train our ChatBot framework.")
    chatbot = ChatBot()
    chatbot.load_data()
    #chatbot.print_docs()
    #chatbot.nltk_download()
    chatbot.get_glove_embeddings()
    chatbot.test_embeddings()
    chatbot.get_counts()
    chatbot.make_embeddings_matrix()
    chatbot.encode_y()
    chatbot.define_model()
    chatbot.train_with_datagen()
    return chatbot

# def save_chatbot_as_pickle(chatbot):
#     filehandler = open("./chatbot_v1", 'wb') 
#     pickle.dump(chatbot, filehandler)
#     filehandler.close()

def predict_from_list(chatbot, lis):
    predicted_disease = chatbot.predict_from_list(lis)
    print(f"\n Predicted disease: {predicted_disease}")

def predict_from_sentence(chatbot, sentence):
    predicted_disease = chatbot.predict_from_sentence(sentence)
    print(f"\n Predicted disease: {predicted_disease}")

def continuous_prediction(chatbot):
    while True:
        lis = [x for x in input("Enter symptoms separated by space and comma: ").split(",")]
        predicted_disease = predict_from_list(chatbot, lis)

if __name__=="__main__":
    chatbot = train_chatbot_first_time()
    #save_chatbot_as_pickle(chatbot)
    lis = ["indigestion",
        "stiff_neck",
        "irritability",
        "headache",
        "acidity"]
    sentence = "i have skin rash dischromic patches itching and nodal skin eruptions"
    predict_from_list(chatbot, lis)
    predict_from_sentence(chatbot, sentence)
    continuous_prediction(chatbot)
    
  
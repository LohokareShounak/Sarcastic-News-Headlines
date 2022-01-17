import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import pandas as pd
import numpy as np

DATA_LOC = r"C:\Users\shounak lohokare\Desktop\Python Basics\Project A\Data\Data.csv"
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
MODEL = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)

print("Preparing Data")
DATA = pd.read_csv(DATA_LOC)

MSK = int(np.round(len(DATA)*0.7))

DATA_TRAIN = DATA.iloc[:MSK]
DATA_TEST = DATA.iloc[MSK:]

sent_train = list(DATA_TRAIN['headline'])
sent_test = list(DATA_TEST['headline'])
label_train = list(DATA_TRAIN['is_sarcastic'])
label_test = list(DATA_TEST['is_sarcastic'])

train_encod = TOKENIZER(sent_train, truncation=True, padding=True)
test_encod = TOKENIZER(sent_test,padding=True,truncation=True)

train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encod),
    label_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encod),
    label_test
))

print("Data Prepared!")

print("Fine Tuning Model")

OPTIMIZER = tf.keras.optimizers.Adam(learning_rate = 5e-5)
MODEL.compile(optimizer = OPTIMIZER, loss = MODEL.compute_loss,
        metrics = ['accuracy']
    )
MODEL.fit(train_dataset.shuffle(100).batch(16),
epochs = 3,
validation_data = test_dataset.shuffle(100).batch(16))

print("Saving Model and Tokenizers")
MODEL.save_pretrained("/Models")
TOKENIZER.save_pretrained("/Models/Tokenizers")


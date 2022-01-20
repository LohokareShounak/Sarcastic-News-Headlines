from transformers import DistilBertForSequenceClassification, BertTokenizer
import torch
import pandas as pd
from transformers import AdamW
import numpy as np
from Data_Preprocess import Data_to_Tensor
from datasets import load_metric
import pickle
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

print("Initialising Variables")
DATA_LOC = r"Data/Data.csv"
TOKENIZER = BertTokenizer.from_pretrained("distilbert-base-uncased")
MODEL = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels = 2)
BATCH_SIZE = 8
DATA = pd.read_csv(DATA_LOC).sample(frac = 1)
MSK = int(np.round(len(DATA)*0.7))
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
OPTIMIZER = AdamW(MODEL.parameters(), lr = 5e-5)
NUM_EPOCHS = 3

###### Changing Device ######
MODEL.to(DEVICE)

###### Preparing Data ######
print("Preparing Data")
input_ids, attention_mask, labels = Data_to_Tensor(DATA, TOKENIZER)
train_input_ids, train_attention_mask, train_labels = input_ids[:MSK], attention_mask[:MSK], labels[:MSK]
test_input_ids, test_attention_mask, test_labels = input_ids[MSK:], attention_mask[MSK:], labels[MSK:]

tensor_data_train = TensorDataset(train_input_ids, train_attention_mask, train_labels)
tensor_data_test = TensorDataset(test_input_ids, test_attention_mask, test_labels)

DATA_TRAIN = DataLoader(tensor_data_train, batch_size=BATCH_SIZE, shuffle=True)
DATA_TEST = DataLoader(tensor_data_test,batch_size=8)


##### Starting Training #####
print("Started Training")

NUM_TRAINING_STEPS = len(DATA_TRAIN)*NUM_EPOCHS

progress_bar = tqdm(range(NUM_TRAINING_STEPS), position=0, leave = True)

MODEL.train()

for epoch in range(NUM_EPOCHS):
    for i, batch in enumerate(DATA_TRAIN):
        #print(f"Epoch - {epoch} Batch - {i}/{len(DATA_TRAIN)}", end = "\r")
        input_ids_tra = batch[0].to(DEVICE)
        attention_mask_tra = batch[1].to(DEVICE)
        labels_tra = batch[2].to(DEVICE)

        output = MODEL(input_ids = input_ids_tra,
                        attention_mask = attention_mask_tra,
                        labels = labels_tra)
        loss = output.loss
        loss.backward()

        OPTIMIZER.step()
        OPTIMIZER.zero_grad()
        progress_bar.update()

print("Training Finish")

print("Saving Model")
MODEL.save_pretrained("Models/Pytorch")
print("Models Saved at - Models/Pytorch")

print("Test Starting")
##### Loading Metrics ######
met = ["accuracy", "recall", "precision", "recall"]
metrics = []
for _ in met:
    metrics.append(load_metric(_))

MODEL.eval()
for batch in DATA_TEST:
    input_ids_tra = batch[0].to(DEVICE)
    attention_mask_tra = batch[1].to(DEVICE)
    labels_tra = batch[2].to(DEVICE)

    output = MODEL(input_ids = input_ids_tra,
                    attention_mask = attention_mask_tra,
                    labels = labels_tra)
    
    predictions = torch.argmax(output.logits, dim = 1)
    for metric in metrics:
        metric.add_batch(predictions=predictions, references=labels_tra)
RESULT = []
for metric in metrics:
    RESULT.append(metric.compute())
print("Saving Results")
with open("Results.p", 'rb') as f:
    pickle.dump(RESULT, f)
print(RESULT)
import torch

def Data_to_Tensor(Data, Tokenizer):
    headlines = Data.headline.values

    #Calculating Max Length of Training Sentences
    max_len = 0
    for sent in headlines:
        if len(sent.split()) > max_len:
            max_len = len(sent.split())

    input_ids = []
    attention_mask = []
    for sent in headlines:
        
        encoded_sent = Tokenizer.encode_plus(sent,
                                max_length = max_len,
                                #return_tensors = "pt",
                                pad_to_max_length = True)

        input_ids.append(encoded_sent.get('input_ids'))
        attention_mask.append(encoded_sent.get('attention_mask'))
    
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(Data.is_sarcastic.values)
    return input_ids, attention_mask, labels

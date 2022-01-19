import streamlit as st
from transformers import DistilBertForSequenceClassification, BertTokenizer
import torch
def Load_Model():
    file_path = r"C:/Users/shounak lohokare/Desktop/Python Basics/Project A/Transformers/Models/Pytorch"
    Model = DistilBertForSequenceClassification.from_pretrained(file_path)
    return Model
def Predict_Page():
    st.title("Headlines these days trying to be funny aren't they?")

    st.write("""
            Lets see if they are or not?
            """)
    
    headline = st.text_input("Headline", value = "")

    if len(headline) > 4:
        Model = Load_Model()

        Tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")

        Text_Encode = Tokenizer.encode_plus(headline,
                        truncation = True,
                        padding = True,
                        return_tensors = "pt")
        input_ids = Text_Encode['input_ids']
        attention_mask = Text_Encode["attention_mask"]
        Output = Model(input_ids, attention_mask)
        print(Output)
        preds = int(torch.argmax(Output.logits))

        if preds == 1:
            st.markdown("![Bazzinga](https://c.tenor.com/0d_WxNIZ6hEAAAAC/big-bang-theory-sheldon-cooper.gif)")
        
        else:
            st.markdown("![:((](https://c.tenor.com/b5jxHCnZHfkAAAAM/the-big-bang-theory-sheldon.gif)")

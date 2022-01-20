import dash_core_components as dcc
import dash_html_components as html
import dash
from dash.dependencies import Output, Input
from transformers import DistilBertForSequenceClassification, BertTokenizer
import torch


app = dash.Dash()
def Load_Model():
    PATH = "../Transformers/Models/Pytorch"
    Model = DistilBertForSequenceClassification.from_pretrained(PATH)
    return Model

app.layout = html.Div(children = 
    [
        html.H1(children="Sarcasm Detection in News Headlines",
                style = {
                    'textAlign' : 'center'
                }),
        html.Br(),
        html.Br(),

        html.H4(children="Headline -"),
        html.Br(),
        html.Br(),

        dcc.Input(
            id = 'headline',
            placeholder= 'Headline',
            type = "text",
            value= ""
        ),

        html.Br(),
        html.Br(),

        html.Div(id = "Output")
    ]
)


@app.callback(
    Output(component_id="Output", component_property="children"),
    [Input(component_id="headline", component_property="value")]
)

def Get_Prediction(headline):
    if len(headline) > 4:
        MODEL = Load_Model()
        TOKENIZER = BertTokenizer.from_pretrained("distilbert-base-uncased")

        text_encode = TOKENIZER.encode_plus(headline, return_tensors = "pt")
        input_ids = text_encode['input_ids']
        attention_mask = text_encode['attention_mask']
        op = MODEL(input_ids, attention_mask)

        preds = int(torch.argmax(op.logits))

        if preds == 1:
            return "Bazinga!!"
        else:
            return "UGHH!!"


if __name__ == "__main__":
    app.run_server(port = 4050)
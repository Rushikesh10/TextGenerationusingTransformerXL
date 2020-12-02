import torch
from pytorch_pretrained_bert import TransfoXLTokenizer,TransfoXLLMHeadModel
from transformers import TransfoXLTokenizer,TransfoXLLMHeadModel
from flask import Flask, request, render_template

def TextGenerator(line):
    tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
    model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    line_tokenized = tokenizer.tokenize(line)
    line_indexed = tokenizer.convert_tokens_to_ids(line_tokenized)
    tokens_tensor = torch.tensor([line_indexed])
    max_predictions = 30
    mems = None
    l=[]
    for i in range(max_predictions):
        predictions, mems = model(tokens_tensor, mems=mems)
        predicted_index_tensor = torch.topk(predictions[0, -1, :], 5)[1][1]
        predicted_index = predicted_index_tensor.item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token)
        l.append(predicted_token)
        tokens_tensor = torch.cat((tokens_tensor, predicted_index_tensor.reshape(1, 1)), dim=1)
        s=" ".join(l)
    return s


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page

def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI

def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            line=str(request.form['line'])
            # predictions using the loaded model file
            pred=TextGenerator(line)
            print(pred)
            # showing the prediction results in a UI
            return render_template('results.html',prediction=str(pred))
        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'
    # return render_template('results.html')
    else:
        return render_template('index.html')


if __name__ == "__main__":
    #app.run(host='127.0.0.1', port=8001, debug=True)
	app.run(debug=True) # running the app


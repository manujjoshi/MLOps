from flask import Flask,request
import pickle

app = Flask(__name__)

classifier = pickle.load(open("knn.pkl","rb"))

@app.route('/')
def home():
    return "heyy! WELCOME TO THE API"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():

      
    
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    if prediction == [0]:
       
       return "the note is not authenticated"
    
    else:

        return "the note is authenticated"
    
    





if __name__=='__main__':
    app.run(debug = True,host='0.0.0.0',port=8080)

    



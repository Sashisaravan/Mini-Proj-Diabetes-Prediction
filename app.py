import flask
import pickle  ## pickle for loading model(Diabetes.pkl)
import pandas as pd  ## to convert the input data into a dataframe for giving as a input to the model


app = flask.Flask(__name__)  ## setting up flask name

model = pickle.load(open("Diabetes.pkl", "rb"))  ##loading model

model2 = pickle.load(open("Diabetessvm.pkl", "rb"))

model3 = pickle.load(open("Diabetesknn.pkl", "rb"))

model4 = pickle.load(open("DiabetesLR.pkl", "rb"))

model5 = pickle.load(open("DiabetesRF.pkl", "rb"))


@app.route('/')             ## Defining main index route
def home():
    return flask.render_template("index.html")   ## showing index.html as homepage

@app.route('/diabetes',methods=['POST','GET'])  ## this route will be called when predict button is called
def diabetes():
    if flask.request.method == 'POST':
        submit_ann = flask.request.form.get('submit_ann', None)
        submit_svm = flask.request.form.get('submit_svm', None)
        submit_knn = flask.request.form.get('submit_knn', None)
        submit_LR = flask.request.form.get('submit_LR', None)
        submit_RF = flask.request.form.get('submit_RF', None)
        if submit_ann:
            text1 = flask.request.form['1']
            text2 = flask.request.form['2']
            text3 = flask.request.form['3']
            text4 = flask.request.form['4']
            text5 = flask.request.form['5']
            text6 = flask.request.form['6']
            text7 = flask.request.form['7']
            text8 = flask.request.form['8']
            row_df = pd.DataFrame([pd.Series([text1,text2,text3,text4,text5,text6,text7,text8])])  ### Creating a dataframe using all the values
            print(row_df)
            prediction = model.predict_proba(row_df) ## Predicting the output
            output='{0:.{1}f}'.format(prediction[0][1], 2)    ## Formating output
            if output>str(0.8):
                return flask.render_template('index.html', pred='Report :  Positive of having diabetes.\n Model used: Neural network. \n Accuracy of ANN model is {}'.format(output)) ## Returning the message for use on the same index.html page
            else:
                return flask.render_template('index.html', pred='Negative of Diabetes.\nModel used: Neural network. \n Accuracy of Ann model is {}'.format(output))

        elif submit_svm:
            text1 = flask.request.form['1']
            text2 = flask.request.form['2']
            text3 = flask.request.form['3']
            text4 = flask.request.form['4']
            text5 = flask.request.form['5']
            text6 = flask.request.form['6']
            text7 = flask.request.form['7']
            text8 = flask.request.form['8']
            row_df = pd.DataFrame([pd.Series([text1, text2, text3, text4, text5, text6, text7, text8])])  ### Creating a dataframe using all the values
            print(row_df)
            prediction = model2.predict_proba(row_df)  ## Predicting the output
            outputSVM = '{0:.{1}f}'.format(prediction[0][1], 2)  ## Formating output
            if outputSVM > str(0.8):
                return flask.render_template('index.html', predsvm='Report : Positive of Diabetes.\nAlgorithm used :Support Vector machine. \nAccuracy of SVM model is {}'.format(outputSVM))  ## Returning the message for use on the same index.html page
            else:
                return flask.render_template('index.html', predsvm='Negative of Diabetes.\nAlgorithm used :Support Vector machine.\n Accuracy of SVM model is {}'.format(outputSVM))

        elif submit_knn:
            text1 = flask.request.form['1']
            text2 = flask.request.form['2']
            text3 = flask.request.form['3']
            text4 = flask.request.form['4']
            text5 = flask.request.form['5']
            text6 = flask.request.form['6']
            text7 = flask.request.form['7']
            text8 = flask.request.form['8']
            row_df = pd.DataFrame([pd.Series([text1, text2, text3, text4, text5, text6, text7,text8])])  ### Creating a dataframe using all the values
            print(row_df)
            prediction = model3.predict_proba(row_df)  ## Predicting the output
            outputknn = '{0:.{1}f}'.format(prediction[0][1], 2)  ## Formating output
            if outputknn > str(0.8):
                return flask.render_template('index.html', predknn='Report : Positive of Diabetes.\nAlgorithm used : K-nearest neighbor .\nAccuracy of KNN model is {}'.format(outputknn))  ## Returning the message for use on the same index.html page
            else:
                return flask.render_template('index.html', predknn='Report : Negative of Diabetes.\n Algorithm used : K-nearest neighbor . \nAccuracy of KNN model is {}'.format(outputknn))

        elif submit_LR:
            text1 = flask.request.form['1']
            text2 = flask.request.form['2']
            text3 = flask.request.form['3']
            text4 = flask.request.form['4']
            text5 = flask.request.form['5']
            text6 = flask.request.form['6']
            text7 = flask.request.form['7']
            text8 = flask.request.form['8']
            row_df = pd.DataFrame([pd.Series([text1, text2, text3, text4, text5, text6, text7,text8])])  ### Creating a dataframe using all the values
            print(row_df)
            prediction = model4.predict_proba(row_df)  ## Predicting the output
            outputLR = '{0:.{1}f}'.format(prediction[0][1], 2)  ## Formating output
            if outputLR > str(0.8):
                return flask.render_template('index.html', predlr='Report : Positive of Diabetes.\nAlgorithm used : Logistics Regression.\nAccuracy of LR model is {}'.format(outputLR))  ## Returning the message for use on the same index.html page
            else:
                return flask.render_template('index.html', predlr='Report : Negative of Diabetes.\n Algorithm used : Logistics Regression . \nAccuracy of LR model is {}'.format(outputLR))

        elif submit_RF:
            text1 = flask.request.form['1']
            text2 = flask.request.form['2']
            text3 = flask.request.form['3']
            text4 = flask.request.form['4']
            text5 = flask.request.form['5']
            text6 = flask.request.form['6']
            text7 = flask.request.form['7']
            text8 = flask.request.form['8']
            row_df = pd.DataFrame([pd.Series([text1, text2, text3, text4, text5, text6, text7,text8])])  ### Creating a dataframe using all the values
            print(row_df)
            prediction = model5.predict_proba(row_df)  ## Predicting the output
            outputRF = '{0:.{1}f}'.format(prediction[0][1], 2)  ## Formating output
            if outputRF > str(0.8):
                return flask.render_template('index.html', predrf='Report : Positive of Diabetes.\nAlgorithm used : Random forest.\nAccuracy of RF model is {}'.format(outputRF))  ## Returning the message for use on the same index.html page
            else:
                return flask.render_template('index.html', predrf='Report : Negative of Diabetes.\n Algorithm used : Random forest . \nAccuracy of RF model is {}'.format(outputRF))


if __name__ == '__main__':
    app.run(debug=True)          ## Running the app as debug==True

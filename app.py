from flask import Flask, render_template, request, flash, redirect
import pickle 
import json
# with open('models/diabetes.pkl', 'rb') as f:
#     diabetes_model = pickle.load(f)
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
# import googlemaps
# # import sklearn
# api_key = 'AIzaSyBvgBXXEvEhJuO9AkCkKhedRqt1lEBffFg'
# gmaps = googlemaps.Client(api_key)
# location = '37.7749,-122.4194'
# radius = 5000
# type = 'doctor'





app = Flask(__name__)

# def predict_diabetes(values):
#     model = pickle.load(open('models/diabetes.pkl','rb'))
#     # values = np.asarray(values)
#     return model.predict(values)[0]

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes (1).pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/cbcCheck", methods=['GET', 'POST'])
def cbcPage():
    return render_template('cbc.html')




@app.route("/healthCheck", methods=['GET', 'POST'])
def healthPage():
    return render_template('healthcheck.html')

#DIABETES---------------------------------------------------------------------------------------

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes_choice.html')

@app.route("/diabetes-manual", methods=['GET', 'POST'])
def diabetesPageManual():
    return render_template('diabetes.html')

@app.route("/diabetes-json", methods=['GET', 'POST'])
def diabetesPageJson():
    return render_template('diabetes_json.html')

@app.route("/diabetes-predict", methods=['POST', 'GET'])
def diabetesPredictPage():
    if request.method == 'POST':
        try:
            if 'json_file' in request.files:
                json_data = request.files['json_file'].read()
                data = json.loads(json_data)
                input_values = []
                
                # Extract values from the JSON data and store in an array
                input_values.append(float(data["pregnancies"]))
                input_values.append(float(data["glucose"]))
                input_values.append(float(data["blood_pressure"]))
                input_values.append(float(data["skin_thickness"]))
                input_values.append(float(data["insulin_level"]))
                input_values.append(float(data["BMI"]))
                input_values.append(float(data["diabetes_pedigree_function"]))
                input_values.append(float(data["age"]))
                
                # Preprocess the input data if required (e.g., convert categorical variables)
                # ...
                
                # Load the pre-trained model
                model = pickle.load(open('models/diabetes (1).pkl','rb'))
                
                values = np.asarray(input_values)
                pred = model.predict(values.reshape(1, -1))[0]
                # Perform prediction using the model
                # input_array = np.array(input_values).reshape(1, -1)
                # pred = model.predict(input_array)[0]
                
                return render_template('predict.html', pred = pred)
            
        except Exception as e:
            message = f"Please upload a JSON file {str(e)}"
            return render_template('diabetes_json.html', message=message)
    
    return render_template('diabetes_choice.html')

#Heart------------------------------------------------------------------------------------------------
@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart_choice.html')

@app.route("/heart-manual", methods=['GET', 'POST'])
def heartPageManual():
    return render_template('heart.html')

@app.route("/heart-json", methods=['GET', 'POST'])
def heartPageJson():
    return render_template('heart_json.html')

@app.route("/heart-predict", methods=['POST', 'GET'])
def heartPredictPage():
    if request.method == 'POST':
        try:
            if 'json_file' in request.files:
                json_data = request.files['json_file'].read()
                data = json.loads(json_data)
                input_values = []
                
                # Extract values from the JSON data and store in an array
                input_values.append(float(data["age"]))
                input_values.append(float(data["sex"]))
                input_values.append(float(data["cp"]))
                input_values.append(float(data["trestbps"]))
                input_values.append(float(data["chol"]))
                input_values.append(float(data["fbs"]))
                input_values.append(float(data["restecg"]))
                input_values.append(float(data["thalach"]))
                input_values.append(float(data["exang"]))
                input_values.append(float(data["oldpeak"]))
                input_values.append(float(data["slope"]))
                input_values.append(float(data["ca"]))
                input_values.append(float(data["thal"]))
                # Preprocess the input data if required (e.g., convert categorical variables)
                # ...
                
                # Load the pre-trained model
                model = pickle.load(open('models/heart.pkl','rb'))
                
                values = np.asarray(input_values)
                pred = model.predict(values.reshape(1, -1))[0]
                # Perform prediction using the model
                # input_array = np.array(input_values).reshape(1, -1)
                # pred = model.predict(input_array)[0]
                
                return render_template('predict.html', pred = pred)
            
        except Exception as e:
            message = f"Please upload a JSON file {str(e)}"
            return render_template('heart_json.html', message=message)
    
    return render_template('heart_choice.html')

#Liver------------------------------------------------------------------------------------------------
@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver_choice.html')

@app.route("/liver-manual", methods=['GET', 'POST'])
def liverPageManual():
    return render_template('liver.html')

@app.route("/liver-json", methods=['GET', 'POST'])
def liverPageJson():
    return render_template('liver_json.html')

@app.route("/liver-predict", methods=['POST', 'GET'])
def liverPredictPage():
    if request.method == 'POST':
        try:
            if 'json_file' in request.files:
                json_data = request.files['json_file'].read()
                data = json.loads(json_data)
                input_values = []
                
                # Extract values from the JSON data and store in an array
                input_values.append(float(data["Age"]))
                input_values.append(float(data["TB"]))
                input_values.append(float(data["DB"]))
                input_values.append(float(data["ALP"]))
                input_values.append(float(data["ALT"]))
                input_values.append(float(data["AST"]))
                input_values.append(float(data["TP"]))
                input_values.append(float(data["ALB"]))
                input_values.append(float(data["AG Ratio"]))
                input_values.append(float(data["Gender"]))
                # Preprocess the input data if required (e.g., convert categorical variables)
                # ...
                
                # Load the pre-trained model
                model = pickle.load(open('models/liver.pkl','rb'))
                
                values = np.asarray(input_values)
                pred = model.predict(values.reshape(1, -1))[0]
                # Perform prediction using the model
                # input_array = np.array(input_values).reshape(1, -1)
                # pred = model.predict(input_array)[0]
                
                return render_template('predict.html', pred = pred)
            
        except Exception as e:
            message = f"Please upload a JSON file {str(e)}"
            return render_template('liver_json.html', message=message)
    
    return render_template('liver_choice.html')


#Kidney------------------------------------------------------------------------------------------------
@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney_choice.html')

@app.route("/kidney-manual", methods=['GET', 'POST'])
def kidneyPageManual():
    return render_template('kidney.html')

@app.route("/kidney-json", methods=['GET', 'POST'])
def kidneyPageJson():
    return render_template('kidney_json.html')

@app.route("/kidney-predict", methods=['POST', 'GET'])
def kidneyPredictPage():
    if request.method == 'POST':
        try:
            if 'json_file' in request.files:
                json_data = request.files['json_file'].read()
                data = json.loads(json_data)
                input_values = []
                
                # Extract values from the JSON data and store in an array
                input_values.append(float(data["age"]))
                input_values.append(float(data["bp"]))
                input_values.append(float(data["al"]))
                input_values.append(float(data["su"]))
                input_values.append(float(data["rbc"]))
                input_values.append(float(data["pc"]))
                input_values.append(float(data["pcc"]))
                input_values.append(float(data["ba"]))
                input_values.append(float(data["bgr"]))
                input_values.append(float(data["bu"]))
                input_values.append(float(data["sc"]))
                input_values.append(float(data["pot"]))
                input_values.append(float(data["wc"]))
                input_values.append(float(data["htn"]))
                input_values.append(float(data["dm"]))
                input_values.append(float(data["cad"]))
                input_values.append(float(data["pe"]))
                input_values.append(float(data["ane"]))
                # Preprocess the input data if required (e.g., convert categorical variables)
                # ...
                
                # Load the pre-trained model
                model = pickle.load(open('models/kidney.pkl','rb'))
                
                values = np.asarray(input_values)
                pred = model.predict(values.reshape(1, -1))[0]
                # Perform prediction using the model
                # input_array = np.array(input_values).reshape(1, -1)
                # pred = model.predict(input_array)[0]
                
                return render_template('predict.html', pred = pred)
            
        except Exception as e:
            message = f"Please upload a JSON file {str(e)}"
            return render_template('kidney_json.html', message=message)
    
    return render_template('kidney_choice.html')


#Breast Cancer------------------------------------------------------------------------------------------------

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')


@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predictPage():
    try:
        if request.method == 'POST':
            to_predict_dict = request.form.to_dict()
            to_predict_list = list(map(float, list(to_predict_dict.values())))
            pred = predict(to_predict_list, to_predict_dict)
    except Exception as e:
        message = f"Please enter valid Data {str(e)}"
        return render_template("home.html", message = message)
    return render_template('predict.html', pred = pred)

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred)




# @app.route("/healthpredict", methods=['POST', 'GET'])
# def healthPredictPage():
#     if request.method == 'POST':
#         try:
#             if 'json_file' in request.files:
#                 json_data = request.files['json_file'].read()
#                 data = json.loads(json_data)
#             # with open('json_file.json', 'r') as f:
#             #      = json.load(f)

#             # Define ranges for each field
#             hgb_ranges = {'male': [13, 17], 'female': [12, 16], 'pregnant': [11, 16]}
#             tlc_range = [4, 11]
#             plt_range = [150000, 450000]
#             neut_range = [2500, 7500]
#             lymph_range = [1000, 4800]
#             eos_range = [0, 500]

#             result = ""
                
#             if data['gender'].lower() == 'male':
#                 if data['hgb'] < hgb_ranges['male'][0]:
#                     result+="Anemia"
#             elif data['gender'].lower() == 'female':
#                 if data['hgb'] < hgb_ranges['female'][0]:
#                     result+="Anemia"
#                 elif data['hgb'] < hgb_ranges['pregnant'][0] and data['pregnant']:
#                     result+="Anemia"
#             if data['tlc'] > tlc_range[1]:
#                 result+="--Infection positive (Leucocytosis)"
#             elif data['tlc'] < tlc_range[0]:
#                 result+="--Immunodeficient (Leucopenia)"
#             if data['plt'] < plt_range[0]:
#                 result+="--Bleeding disorder (Thrombocytopenia)"
#             elif data['plt'] > plt_range[1]:
#                 result+="--Thrombocytosis"
#             if data['neutrophile'] > neut_range[1]:
#                 result+="--Neutrocytosis (Bacterial infection)"
#             if data['lymphocyte'] > lymph_range[1]:
#                 result+="--Lymphocytosis (Viral infection)"
#             if data['eosinophil'] > eos_range[1]:
#                 result+="--Allergic reaction"
                
#             return render_template('cbc_predict.html', result = result)


#         except Exception as e:
#             message = f"Please upload a JSON file {str(e)}"
#             return render_template('healthcheck.html', message=message)
    
#     return render_template('heart.html')

# @app.route("/healthpredict", methods=['POST', 'GET'])
# def healthPredictPage():
#     if request.method == 'POST':
#         try:
#             if 'json_file' in request.files:
#                 json_data = request.files['json_file'].read()
#                 data = json.loads(json_data)
                
#                 # Extract values from the JSON data
#                 hgb = float(data["hgb"])
#                 tlc = float(data["tlc"])
#                 plt = float(data["plt"])
#                 neutrophile = float(data["neutrophile"])
#                 lymphocyte = float(data["lymphocyte"])
#                 eosinophil = float(data["eosinophil"])
                
#                 # Check the values against the given ranges
#                 result = ""
#                 if hgb < 13.0:
#                     if data["gender"].lower() == "male":
#                         result += "Anemia (male)<br>"
#                     elif data["gender"].lower() == "female":
#                         if not data["pregnant"].lower() == "true" and hgb < 12.0:
#                             result += "Anemia (non-pregnant female)<br>"
#                         elif data["pregnant"].lower() == "true" and hgb < 11.0:
#                             result += "Anemia (pregnant female)<br>"
                
#                 if tlc < 4000:
#                     result += "Immunodeficiency (leucopenia)<br>"
#                 elif tlc > 11000:
#                     result += "Infection (leucocytosis)<br>"
                
#                 if plt < 150000:
#                     result += "Bleeding disorder (thrombocytopenia)<br>"
#                 elif plt > 450000:
#                     result += "Thrombocytosis<br>"
                
#                 if neutrophile > 7500:
#                     result += "Bacterial infection (neutrocytosis)<br>"
                
#                 if lymphocyte > 4800:
#                     result += "Viral infection (lymphocytosis)<br>"
                
#                 if eosinophil > 500:
#                     result += "Allergic reaction (eosinophilia)<br>"
                
#                 return render_template('cbc_predict.html', result=result)
            
#         except Exception as e:
#             message = f"Please upload a JSON file {str(e)}"
#             return render_template('healthcheck.html', message=message)
    
#     return render_template('healthcheck.html')



# @app.route("/healthpredict", methods=['POST', 'GET'])
# def healthPredictPage():
#     if request.method == 'POST':
#         try:
#             if 'json_file' in request.files:
#                 json_data = request.files['json_file'].read()
#                 data = json.loads(json_data)
#                 input_values = []
                
#                 # Extract values from the JSON data and store in an array
#                 input_values.append(float(data["sex"]))
#                 input_values.append(float(data["pregnancies"]))
#                 input_values.append(float(data["glucose"]))
#                 input_values.append(float(data["blood_pressure"]))
#                 input_values.append(float(data["skin_thickness"]))
#                 input_values.append(float(data["insulin_level"]))
#                 input_values.append(float(data["BMI"]))
#                 input_values.append(float(data["diabetes_pedigree_function"]))
#                 input_values.append(float(data["age"]))
                
#                 # Preprocess the input data if required (e.g., convert categorical variables)
#                 # ...
                
#                 # Load the pre-trained model
#                 model = pickle.load(open('models/diabetes.pkl','rb'))
                
#                 values = np.asarray(input_values)
#                 pred = model.predict(values.reshape(1, -1))[0]
#                 # Perform prediction using the model
#                 # input_array = np.array(input_values).reshape(1, -1)
#                 # pred = model.predict(input_array)[0]
                
#                 return render_template('predict.html', pred = pred)
            
#         except Exception as e:
#             message = f"Please upload a JSON file {str(e)}"
#             return render_template('healthcheck.html', message=message)
    
#     return render_template('heart.html')

# @app.route("/healthpredict", methods=['POST', 'GET'])
# def healthPredictPage():
#     if request.method == 'POST':
#         try:
#             if 'json_file' in request.files:
#                 json_data = request.files['json_file'].read()
#                 data = json.loads(json_data)
#                 input_values = []
                
#                 # Extract values from the JSON data and store in an array
#                 input_values.append(float(data["Age"]))
#                 input_values.append(float(data["TB"]))
#                 input_values.append(float(data["DB"]))
#                 input_values.append(float(data["ALP"]))
#                 input_values.append(float(data["ALT"]))
#                 input_values.append(float(data["AST"]))
#                 input_values.append(float(data["TP"]))
#                 input_values.append(float(data["ALB"]))
#                 input_values.append(float(data["AG Ratio"]))
#                 input_values.append(float(data["Gender"]))
                
#                 # Preprocess the input data if required (e.g., convert categorical variables)
#                 # ...
                
#                 # Load the pre-trained model
#                 model = pickle.load(open('models/liver.pkl','rb'))
                
#                 values = np.asarray(input_values)
#                 pred = model.predict(values.reshape(1, -1))[0]
#                 # Perform prediction using the model
#                 # input_array = np.array(input_values).reshape(1, -1)
#                 # pred = model.predict(input_array)[0]
                
#                 return render_template('predict.html', pred = pred)
            
#         except Exception as e:
#             message = f"Please upload a JSON file {str(e)}"
#             return render_template('healthcheck.html', message=message)
    
#     return render_template('heart.html')

# @app.route("/healthpredict", methods=['POST', 'GET'])
# def healthPredictPage():
#     if request.method == 'POST':
#         try:
#             if 'json_file' in request.files:
#                 json_data = request.files['json_file'].read()
#                 data = json.loads(json_data)
#                 input_values = []
                
#                 # Extract values from the JSON data and store in an array
#                 input_values.append(float(data["age"]))
#                 input_values.append(float(data["sex"]))
#                 input_values.append(float(data["cp"]))
#                 input_values.append(float(data["trestbps"]))
#                 input_values.append(float(data["chol"]))
#                 input_values.append(float(data["fbs"]))
#                 input_values.append(float(data["restecg"]))
#                 input_values.append(float(data["thalach"]))
#                 input_values.append(float(data["exang"]))
#                 input_values.append(float(data["oldpeak"]))
#                 input_values.append(float(data["slope"]))
#                 input_values.append(float(data["ca"]))
#                 input_values.append(float(data["thal"]))
#                 # Preprocess the input data if required (e.g., convert categorical variables)
#                 # ...
                
#                 # Load the pre-trained model
#                 model = pickle.load(open('models/heart.pkl','rb'))
                
#                 values = np.asarray(input_values)
#                 pred = model.predict(values.reshape(1, -1))[0]
#                 # Perform prediction using the model
#                 # input_array = np.array(input_values).reshape(1, -1)
#                 # pred = model.predict(input_array)[0]
                
#                 return render_template('predict.html', pred = pred)
            
#         except Exception as e:
#             message = f"Please upload a JSON file {str(e)}"
#             return render_template('healthcheck.html', message=message)
    
#     return render_template('heart.html')

# @app.route("/healthpredict", methods=['POST', 'GET'])
# def healthPredictPage():
#     if request.method == 'POST':
#         try:
#             if 'json_file' in request.files:
#                 json_data = request.files['json_file'].read()
#                 data = json.loads(json_data)
#                 input_values = []
                
#                 # Extract values from the JSON data and store in an array
#                 input_values.append(float(data["age"]))
#                 input_values.append(float(data["bp"]))
#                 input_values.append(float(data["al"]))
#                 input_values.append(float(data["su"]))
#                 input_values.append(float(data["rbc"]))
#                 input_values.append(float(data["pc"]))
#                 input_values.append(float(data["pcc"]))
#                 input_values.append(float(data["ba"]))
#                 input_values.append(float(data["bgr"]))
#                 input_values.append(float(data["bu"]))
#                 input_values.append(float(data["sc"]))
#                 input_values.append(float(data["pot"]))
#                 input_values.append(float(data["wc"]))
#                 input_values.append(float(data["htn"]))
#                 input_values.append(float(data["dm"]))
#                 input_values.append(float(data["cad"]))
#                 input_values.append(float(data["pe"]))
#                 input_values.append(float(data["ane"]))
#                 # Preprocess the input data if required (e.g., convert categorical variables)
#                 # ...
                
#                 # Load the pre-trained model
#                 model = pickle.load(open('models/kidney.pkl','rb'))
                
#                 values = np.asarray(input_values)
#                 pred = model.predict(values.reshape(1, -1))[0]
#                 # Perform prediction using the model
#                 # input_array = np.array(input_values).reshape(1, -1)
#                 # pred = model.predict(input_array)[0]
                
#                 return render_template('predict.html', pred = pred)
            
#         except Exception as e:
#             message = f"Please upload a JSON file {str(e)}"
#             return render_template('healthcheck.html', message=message)
    
#     return render_template('heart.html')

@app.route("/healthpredict", methods=['POST', 'GET'])
def healthPredictPage():
    if request.method == 'POST':
        try:
            if 'json_file' in request.files:
                json_data = request.files['json_file'].read()
                data = json.loads(json_data)
                common_values = []
                # cbc_values = []
                diabetes_value = []
                liver_value = []
                heart_value = []
                kidney_value = []

                #common
                # common_values.append(string(data["name"]))
                common_values.append(float(data["age"]))
                common_values.append(float(data["sex"]))
                
                #cbc-test
                #gender = float(data["sex"])
                hgb = float(data["hgb"])
                tlc = float(data["tlc"])
                plt = float(data["plt"])
                neutrophile = float(data["neutrophile"])
                lymphocyte = float(data["lymphocyte"])
                eosinophil = float(data["eosinophil"])
                
                # Check the values against the given ranges
                result = ""
                pred_anemia = 0
                pred_leucopenia =0
                pred_leucocytosis = 0
                pred_thrombocytopenia = 0
                pred_Thrombocytosis = 0
                pred_neutrocytosis = 0
                pred_lymphocytosis = 0
                pred_eosinophilia = 0
                if hgb < 13.0:
                    if data["sex"] == 1:
                        pred_anemia = 1
                        #result += "Anemia (male)<br>"
                    elif data["sex"] == 0:
                        if not data["pregnancies"] == 0 and hgb < 12.0:
                            pred_anemia = 1
                            #result += "Anemia (non-pregnant female)<br>"
                        elif data["pregnancies"] > 0 and hgb < 11.0:
                            pred_anemia = 1
                            #result += "Anemia (pregnant female)<br>"
                
                if tlc < 4000:
                    pred_leucopenia = 1
                    #result += "Immunodeficiency (leucopenia)<br>"
                elif tlc > 11000:
                    pred_leucocytosis = 1
                    #result += "Infection (leucocytosis)<br>"
                
                if plt < 150000:
                    pred_thrombocytopenia = 1
                    #result += "Bleeding disorder (thrombocytopenia)<br>"
                elif plt > 450000:
                    pred_Thrombocytosis = 1
                    #result += "Thrombocytosis<br>"
                
                if neutrophile > 7500:
                    pred_neutrocytosis = 1
                    #result += "Bacterial infection (neutrocytosis)<br>"
                
                if lymphocyte > 4800:
                    pred_lymphocytosis = 1
                    #result += "Viral infection (lymphocytosis)<br>"
                
                if eosinophil > 500:
                    pred_eosinophilia = 1
                    #result += "Allergic reaction (eosinophilia)<br>"
                
                #render_template('cbc_predict.html', result=result)
            
                #diabetes
                # diabetes_value.append(float(data["sex"]))
                diabetes_value.append(float(data["pregnancies"]))
                diabetes_value.append(float(data["glucose"]))
                diabetes_value.append(float(data["blood_pressure"]))
                diabetes_value.append(float(data["skin_thickness"]))
                diabetes_value.append(float(data["insulin_level"]))
                diabetes_value.append(float(data["BMI"]))
                diabetes_value.append(float(data["diabetes_pedigree_function"]))
                diabetes_value.append(float(data["age"]))
                model = pickle.load(open('models/diabetes (1).pkl','rb'))
                values = np.asarray(diabetes_value)
                pred_dia = model.predict(values.reshape(1, -1))[0]
                # return render_template('predict.html', pred = pred)
            
                #liver

                liver_value.append(float(data["age"]))
                liver_value.append(float(data["TB"]))
                liver_value.append(float(data["DB"]))
                liver_value.append(float(data["ALP"]))
                liver_value.append(float(data["ALT"]))
                liver_value.append(float(data["AST"]))
                liver_value.append(float(data["TP"]))
                liver_value.append(float(data["ALB"]))
                liver_value.append(float(data["AG Ratio"]))
                liver_value.append(float(data["sex"]))
                model = pickle.load(open('models/liver.pkl','rb'))
                values = np.asarray(liver_value)
                pred_liver = model.predict(values.reshape(1, -1))[0]
                #return render_template('predict.html', pred = pred)
                #heart
                heart_value.append(float(data["age"]))
                heart_value.append(float(data["sex"]))
                heart_value.append(float(data["cp"]))
                heart_value.append(float(data["blood_pressure"]))
                heart_value.append(float(data["chol"]))
                heart_value.append(float(data["fbs"]))
                heart_value.append(float(data["restecg"]))
                heart_value.append(float(data["thalach"]))
                heart_value.append(float(data["exang"]))
                heart_value.append(float(data["oldpeak"]))
                heart_value.append(float(data["slope"]))
                heart_value.append(float(data["ca"]))
                heart_value.append(float(data["thal"]))
                model = pickle.load(open('models/heart.pkl','rb'))
                values = np.asarray(heart_value)
                pred_heart = model.predict(values.reshape(1, -1))[0]   
                #return render_template('predict.html', pred = pred)

                #kidney
                kidney_value.append(float(data["age"]))
                kidney_value.append(float(data["blood_pressure"]))
                kidney_value.append(float(data["ALB"]))
                kidney_value.append(float(data["su"]))
                kidney_value.append(float(data["rbc"]))
                kidney_value.append(float(data["pc"]))
                kidney_value.append(float(data["pcc"]))
                kidney_value.append(float(data["ba"]))
                kidney_value.append(float(data["bgr"]))
                kidney_value.append(float(data["bu"]))
                kidney_value.append(float(data["sc"]))
                kidney_value.append(float(data["pot"]))
                kidney_value.append(float(data["wc"]))
                kidney_value.append(float(data["htn"]))
                kidney_value.append(float(data["dm"]))
                kidney_value.append(float(data["cad"]))
                kidney_value.append(float(data["pe"]))
                kidney_value.append(float(data["ane"]))
                model = pickle.load(open('models/kidney.pkl','rb'))
                values = np.asarray(kidney_value)
                pred_kidney = model.predict(values.reshape(1, -1))[0]

                report = [pred_anemia, pred_leucopenia ,pred_leucocytosis , pred_thrombocytopenia, pred_Thrombocytosis, pred_neutrocytosis, pred_lymphocytosis, pred_eosinophilia, pred_dia,pred_liver,pred_heart,pred_kidney]
                return render_template('report.html', report = report, result = result)
                #return render_template('predict.html', pred = pred)
            
        except Exception as e:
            message = f"Please upload a JSON file {str(e)}"
            return render_template('healthcheck.html', message=message)
    
    return render_template('heart.html')

@app.route("/cbc-predict", methods=['POST', 'GET'])
def cbcPredictPage():
    if request.method == 'POST':
        try:
            if 'json_file' in request.files:
                json_data = request.files['json_file'].read()
                data = json.loads(json_data)
                input_values = []
                
                # Extract values from the JSON data and store in an array
                hgb = float(data["hgb"])
                tlc = float(data["tlc"])
                plt = float(data["plt"])
                neutrophile = float(data["neutrophile"])
                lymphocyte = float(data["lymphocyte"])
                eosinophil = float(data["eosinophil"])
                # Preprocess the input data if required (e.g., convert categorical variables)
                # ...
                pred_anemia = 0
                pred_leucopenia =0
                pred_leucocytosis = 0
                pred_thrombocytopenia = 0
                pred_Thrombocytosis = 0
                pred_neutrocytosis = 0
                pred_lymphocytosis = 0
                pred_eosinophilia = 0

                if hgb < 13.0:
                    if data["sex"] == 1:
                        pred_anemia = 1
                        #result += "Anemia (male)<br>"
                    elif data["sex"] == 0:
                        if not data["pregnancies"] == 0 and hgb < 12.0:
                            pred_anemia = 1
                            #result += "Anemia (non-pregnant female)<br>"
                        elif data["pregnancies"] > 0 and hgb < 11.0:
                            pred_anemia = 1
                            #result += "Anemia (pregnant female)<br>"
                
                if tlc < 4000:
                    pred_leucopenia = 1
                    #result += "Immunodeficiency (leucopenia)<br>"
                elif tlc > 11000:
                    pred_leucocytosis = 1
                    #result += "Infection (leucocytosis)<br>"
                
                if plt < 150000:
                    pred_thrombocytopenia = 1
                    #result += "Bleeding disorder (thrombocytopenia)<br>"
                elif plt > 450000:
                    pred_Thrombocytosis = 1
                    #result += "Thrombocytosis<br>"
                
                if neutrophile > 7500:
                    pred_neutrocytosis = 1
                    #result += "Bacterial infection (neutrocytosis)<br>"
                
                if lymphocyte > 4800:
                    pred_lymphocytosis = 1
                    #result += "Viral infection (lymphocytosis)<br>"
                
                if eosinophil > 500:
                    pred_eosinophilia = 1
                # Perform prediction using the model
                # input_array = np.array(input_values).reshape(1, -1)
                # pred = model.predict(input_array)[0]
                report = [pred_anemia, pred_leucopenia ,pred_leucocytosis , pred_thrombocytopenia, pred_Thrombocytosis, pred_neutrocytosis, pred_lymphocytosis, pred_eosinophilia]
                return render_template('cbc_report.html', report = report)
            
        except Exception as e:
            message = f"Please upload a JSON file {str(e)}"
            return render_template('cbc.html', message=message)
    
    return render_template('cbc.html')


if __name__ == '__main__':
	app.run(debug = True)
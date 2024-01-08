import pickle
import csv
import json

from flask import Flask, request, jsonify, render_template, send_from_directory, Blueprint
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import yeojohnson

np.random.seed(42)

class CustomNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, lmbda=None):
        self.lmbda = lmbda
    
    def fit(self, X, y=None):
        # If lambda is not provided, estimate it for each feature
        if self.lmbda is None:
            self.lmbda_ = []
            for feature in X.T:
                transformed_feature, lmbda = yeojohnson(feature)
                self.lmbda_.append(lmbda)                
        else:
            self.lmbda_ = [self.lmbda] * X.shape[1]
        return self
    
    def transform(self, X):
        # Apply the Box-Cox transformation to each feature
        normalized_X = np.zeros_like(X)
        
        for i, (feature, lmbda) in enumerate(zip(X.T, self.lmbda_)):
            normalized_X[:, i] = yeojohnson(feature, lmbda=lmbda)
        return normalized_X

app = Flask(__name__)
CORS(app)

prefix = '/geoviz'
blueprint = Blueprint('blueprint', __name__, url_prefix=prefix)

reducers = {}
classifiers = {}
scalers = {}

for i in range(3):
    files = open("models/reducers/reducer{}.pkl".format(i), 'rb')
    reducers[i] = pickle.load(files)

    files = open("models/classifiers/classifier{}.pkl".format(i), 'rb')
    classifiers[i] = pickle.load(files)

    files = open("models/scalers/scaler{}.pkl".format(i), 'rb')
    scalers[i] = pickle.load(files)


@blueprint.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        # Process the uploaded file
        # You can save it, read it, or perform any other desired actions
        data = file.read().decode('utf-8')
        
        csv_data = list(csv.reader(data.splitlines()[1:]))

        df1 = pd.DataFrame(csv_data, columns=data.splitlines()[0].split(","))
        
        df1.fillna(0, inplace=True)
        columns = ['TiO2', 'Cr', 'Ti', 
            'Al', 'Fe3+', 'Fe2+',  'Mg',
            'Cr2O3', 'Al2O3', 'FeO', 'Fe2O3', 'MgO']
        em = ["MgAl2O4", "FeAl2O4", "MgFe2O4", "FeFe2O4", "MgCr2O4", "FeCr2O4", "Mg2TiO4", "Fe2TiO4"]

        columns = columns + em
        df1 = df1[columns]

        df1 = df1.astype('float32')

        df1["numfe2"] = df1['Fe2+']/(df1['Fe2+']+df1.Mg)
        df1["numcr"] = df1.Cr / (df1.Cr + df1.Al)
        df1["numfe3"] = df1['Fe3+'] / (df1['Fe3+']+df1.Cr+df1.Al)
        df1["numti"] = df1.Ti / (df1.Ti + df1.Cr + df1.Al)

        df1.fillna(0, inplace=True)

        Q_setting1 = [
            "Ti", "Cr", "Al", "Fe2+", "Fe3+", "Mg",
            "TiO2", 'Cr2O3', 'Al2O3', 'FeO', 'Fe2O3', 'MgO', 
            "numfe2", "numcr","numfe3"]

        Q_setting2 = [
            "Ti", "Cr", "Al", "Fe2+", "Fe3+", "Mg",
            "TiO2", 'Cr2O3', 'Al2O3', 'FeO', 'Fe2O3', 'MgO', 
            "numti", "numfe2", "numcr","numfe3"]

        Q_values_nuevo1 = df1[Q_setting1].values
        Q_values_nuevo2 = df1[Q_setting2].values
        Q_values_nuevo3 = df1[em].values

        Q_values_nuevo = [Q_values_nuevo1, Q_values_nuevo2, Q_values_nuevo3]


        i = int(request.form.get('option'))+1

        scaler = scalers[i]
        clf = classifiers[i]
        red = reducers[i]

        Q_values = Q_values_nuevo[i]
        
        xval = scaler.transform(Q_values)
        leaves = clf.apply(xval)
        XVproj = red.transform(leaves)

        #result = pd.DataFrame({'umap1':XVproj[:,0], 'umap2':XVproj[:,1]}).to_json()

        classes = ['ALASKAN ULTRAMAFIC', 'ALKALI/LAMPROPHYRES', 'BASALT', 'KOMATIITE',
       'LAYERED INTRUSION', 'METAMORPHIC', 'OPHIOLITE', 'XENOLITHS']

        if (i==2):
            result = pd.DataFrame(np.c_[Q_values_nuevo2, Q_values,clf.predict_proba(xval), df1[em].values, XVproj], 
                columns=Q_setting2+em+classes+em+["umap1", "umap2"])
        elif (i==0):
            result = pd.DataFrame(np.c_[Q_values,clf.predict_proba(xval), df1[em].values, XVproj], 
                columns=Q_setting1+classes+em+["umap1", "umap2"])
        else:
            result = pd.DataFrame(np.c_[Q_values,clf.predict_proba(xval), df1[em].values, XVproj], 
                columns=Q_setting2+classes+em+["umap1", "umap2"])


        result = result.to_dict(orient='records')

        return json.dumps(result)

# @blueprint.route('/')
# def index():
#     return "hola"

# # Ruta para la página principal
@blueprint.route('/')
def index():
    return render_template('index.html')


# # Ruta para servir archivos estáticos (JS, CSS, imágenes, etc.)
@blueprint.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)




app.register_blueprint(blueprint)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5501, debug=True) #, ssl_context=('cert.pem', 'key.pem'))
    #app.run(host='0.0.0.0', port=8500, debug=True) #, ssl_context=('cert.pem', 'key.pem'))

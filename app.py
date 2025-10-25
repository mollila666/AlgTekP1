from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import sys
import numpy as np
from PIL import Image
#ohjelma on testausta varten ./KNN/src/knn/ hakemistossa
sys.path.append('KNN/src/knn')
from test_class import KNN

app = Flask(__name__)
#Kuvat hakemistoon kuvat_hakemisto
app.config['UPLOAD_FOLDER'] = 'kuvat_hakemisto'
#Sallitut kuvat: .png, .jpg ja .jpeg
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

#Jos hakemistoa kuvat_hakemisto ei ole, se luodaan
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #Tallennetaan valittu kuva
        tiedosto = request.files['tiedosto']
        tiedosto_nimi = tiedosto.filename
        tiedosto_polku = os.path.join(app.config['UPLOAD_FOLDER'], tiedosto_nimi)
        tiedosto.save(tiedosto_polku)
        #Etaisyysmitan valinta
        mitta1 = 'mitta1' in request.form
        mitta2 = 'mitta2' in request.form
        mitta3 = 'mitta3' in request.form
        mitta4 = 'mitta4' in request.form
        #Haetaan kuva
        kuva = Image.open(tiedosto_polku).convert("L")  
        #Kuvan pikselien lukumaara
        kuva = kuva.resize((28, 28))
        #Skaalataan pikseleiden arvot valille 0 - 1
        kuva=np.array(kuva)/255.0
        #Pikseleiden arvo jo 0 tai 1, mustavalkoinen
        kuva = (kuva > 0.5).astype(np.float32)
        #Euklidinen etaisyysmitta
        if mitta1:
            mitta='euklidinen'
        #d22 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti
        elif mitta2:
            mitta='d22'
        #d23 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti
        elif mitta3:
            mitta='d23'
        #d23 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti,
        #ilman keskiarvoistusta etaisyydessa d6
        elif mitta4:
            mitta='d23_summa'
        #Euklidinen etaisyysmitta, mikali etaisyysmittaa ei valittu
        else:
            mitta='euklidinen'
        malli=KNN(etaisyys=mitta, nr_k=7, koulutuskuva_nr=4000)
        tulos = malli.tunnista(kuva)
        return render_template('index.html', tiedosto_nimi=tiedosto_nimi, mitta=mitta, tulos=tulos)
    return render_template('index.html', tiedosto_nimi=None, tulos=None)

@app.route('/uploads/<tiedosto_nimi>')
def uploaded_file(tiedosto_nimi):
    return send_from_directory(app.config['UPLOAD_FOLDER'], tiedosto_nimi)

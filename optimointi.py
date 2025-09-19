#Matematiikka
import numpy as np
#MNIST tietokanta
import tensorflow as tf
#Aika
import time

(x_koulutus, y_koulutus), (x_testi, y_test) = tf.keras.datasets.mnist.load_data()
#Skaalataan pikseleiden arvot valille 0 - 1
x_koulutus=x_koulutus/255.0
x_testi=x_testi/255.0
#Pikseleiden arvo jo 0 tai 1, mustavalkoinen
x_koulutus = (x_koulutus > 0.5).astype(np.float32)
x_testi = (x_testi > 0.5).astype(np.float32)

#tunnistettava
x = x_testi[0]
x_coord = np.argwhere(x)

# Haetaan kaikki koordinaatit valmiiksi
x_koulutus_coord = [np.argwhere(img) for img in x_koulutus]

#alueen koko, lahialue boolean-listasta
koko=5
raja1=int(koko/2)
raja2=raja1+1
pikseli_lkm=28

#Koulutuskuvien lkm
k_lkm=10000

kaikki_etaisyydet_1 = []
t1 = time.time()
for i in range(k_lkm):
    X = x_koulutus[i]
    X_coord = x_koulutus_coord[i]
    etaisyydet = []
    for y_x, x_x in x_coord:
        #luodaan alue jota tutkitaan
        y0, y1 = max(0, y_x - raja1), min(pikseli_lkm, y_x + raja2)
        x0, x1 = max(0, x_x - raja1), min(pikseli_lkm, x_x + raja2)
        alue = X[y0:y1, x0:x1]
        #pisteen lahialue boolean-listasta
        if alue.any():
            #valkoiset pikselit alueesta
            alue_pikselit = np.argwhere(alue)
            #Koordinaatiston muunnos
            alue_pikselit[:, 0] += y0
            alue_pikselit[:, 1] += x0
            #Etaisyydet
            etaisyys = np.linalg.norm(alue_pikselit - np.array([y_x, x_x]), axis=1)
            etaisyydet.append(np.min(etaisyys))
        else:
            #käydaan lapi toisen kuvan koordinaattilista
            etaisyys = np.linalg.norm(X_coord - np.array([y_x, x_x]), axis=1)
            etaisyydet.append(np.min(etaisyys))
    kaikki_etaisyydet_1.append(np.linalg.norm(etaisyydet))

t_1=time.time() - t1

#pudotetaan pisteen lähialueen lapikayminen boolean-listasta pois
kaikki_etaisyydet_2 = []
t1 = time.time()
for i in range(k_lkm):
    X = x_koulutus[i]
    X_coord = x_koulutus_coord[i]
    etaisyydet = []
    for y_x, x_x in x_coord:
        #käydaan lapi toisen kuvan koordinaattilista
        etaisyys = np.linalg.norm(X_coord - np.array([y_x, x_x]), axis=1)
        etaisyydet.append(np.min(etaisyys))
    kaikki_etaisyydet_2.append(np.linalg.norm(etaisyydet))

t_2=time.time() - t1

#Pudotetaan toinen for luuppi pois ja kaytetaan vain vektorisointia
kaikki_etaisyydet_3 = []
t1 = time.time()
for i in range(k_lkm):
    X_coord = x_koulutus_coord[i]
    etaisyys = np.linalg.norm(x_coord[:, None] - X_coord[None, :], axis=2)
    minimi_etaisyys = np.min(etaisyys, axis=1)  
    kaikki_etaisyydet_3.append(np.linalg.norm(minimi_etaisyys))

t_3=time.time() - t1

#pisteen lähialueen lapikayminen boolean-listasta & käydaan lapi toisen kuvan koordinaattilista
t_1
#käydaan lapi vain toisen kuvan koordinaattilista
t_2
#Pudotetaan toinen for luuppi pois ja kaytetaan vain vektorisointia
t_3

t_1/t_2
t_2/t_3
t_1/t_3

np.array(kaikki_etaisyydet_3)-np.array(kaikki_etaisyydet_2)

np.array(kaikki_etaisyydet_1)-np.array(kaikki_etaisyydet_2)
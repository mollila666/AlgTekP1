"""
KNN moduli     
"""
#Matematiikka
import numpy as np
#MNIST tietokanta
import tensorflow as tf

class KNN:
    """
    KNN luokka tunnistaa kasin kirjoitettuja numeroita
    luokalle voidaan maarittaa seuraavat parametrit: 
    etaisyys = etaisyysmitta ja nr_k = naapurien lukumaara 
    """
    def __init__(self, etaisyys='euklidinen', nr_k=5):
        self.etaisyys = etaisyys
        self.nr_k = nr_k
        #Ladataan MNIST tietokannasta data
        (self.x_koulutus, self.y_koulutus) = tf.keras.datasets.mnist.load_data()[0]
        #Skaalataan pikseleiden arvot valille 0 - 1
        self.x_koulutus=self.x_koulutus/255.0
        #Pikseleiden arvo jo 0 tai 1, mustavalkoinen
        self.x_koulutus = (self.x_koulutus > 0.5).astype(np.float32)
        #Matriisit vektoreiksi Euklidisen etaisyyden laskentaa varten
        self.x_koulutus_tasainen = self.x_koulutus.reshape(len(self.x_koulutus), -1)
    def euklidinen_etaisyys(self, x, xmnist):
        """
        Euklidinen etaisyysmitta, lasketaan matriisin X rivien ja vektorin x valiset etaisyydet
        """
        #Muutetaan matriisi x, koko (28,28), vektoriksi, koko (784)
        x = x.reshape(-1)
        #Palautetaan matriisin xmnist rivien ja vektorin x valiset etaisyydet
        return np.linalg.norm(xmnist - x, axis=1)
    def tunnista(self, x):
        """
        Tunnistamis funktio
        """
        if self.nr_k < 1:
            raise ValueError("Naapureiden lukumaara pitaa olla vahintaan 1kpl!")
        if len(x) != 28 or len(x[0]) != 28:
            raise ValueError("Kuvan pikseleiden lukumaara ei ole oikea!")
        if not np.all(np.isin(x, [0.0, 1.0])):
            raise ValueError("Kuvan pitaa olla mustavalkoinen!")
        if self.etaisyys == 'euklidinen':
            #Lasketaan etaisyydet
            etaisyydet = self.euklidinen_etaisyys(x, self.x_koulutus_tasainen)
            #nr_k lahinta naapuria
#            k_indeksit = np.argsort(etaisyydet)[:self.nr_k]
            k_indeksit = np.argpartition(etaisyydet, self.nr_k)[:self.nr_k]
            #nr_k lahinta naapuria numerot
            k_leimat = [self.y_koulutus[i] for i in k_indeksit]
        else:
            raise ValueError("Maarita etaisyysmitta oikein!")
        #Palautetaan numero joka esiintyy useimmin lahimpien naapurien joukkossa
        return np.bincount(k_leimat).argmax()

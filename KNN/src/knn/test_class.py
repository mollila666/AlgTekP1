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
    def __init__(self, etaisyys='euklidinen', nr_k=7, n_karkiehdokkaat=1000):
        self.etaisyys = etaisyys
        self.nr_k = nr_k
        self.n_karkiehdokkaat = n_karkiehdokkaat
        #Ladataan MNIST tietokannasta data
        (self.x_koulutus, self.y_koulutus) = tf.keras.datasets.mnist.load_data()[0]
        #Skaalataan pikseleiden arvot valille 0 - 1
        self.x_koulutus=self.x_koulutus/255.0
        #Pikseleiden arvo jo 0 tai 1, mustavalkoinen
        self.x_koulutus = (self.x_koulutus > 0.5).astype(np.float32)
        #Matriisit vektoreiksi Euklidisen etaisyyden laskentaa varten
        self.x_koulutus_tasainen = self.x_koulutus.reshape(len(self.x_koulutus), -1)
    def euklidinen_etaisyys(self, kuva_1, kuva_2):
        """
        Euklidinen etaisyysmitta, lasketaan pisteiden valiset etaisyydet
        """
        dists = np.linalg.norm(kuva_1[:, None] - kuva_2[None, :], axis=2)
        min_dists = np.min(dists, axis=1)
        return np.linalg.norm(min_dists)
    def d6(self, kuva_1, kuva_2):
        """
        d6 viitteen (Dubuisson & Jain, 1994) mukaisesti
        """
        #Lasketaan kaikki etaisyydet kuvien kuva_1 ja kuva_2 pikseleiden valilla
        dists = np.linalg.norm(kuva_1[:, None] - kuva_2[None, :], axis=2)
        #Minimit
        min_dists = np.min(dists, axis=1)
        #Palautetaan keskiarvo
        return np.mean(min_dists)
    def d6_summa(self, kuva_1, kuva_2):
        """
        d6 viitteen (Dubuisson & Jain, 1994) mukaisesti, ilman keskiarvoistusta
        """
        #Lasketaan kaikki etaisyydet kuvien kuva_1 ja kuva_2 pikseleiden valilla
        dists = np.linalg.norm(kuva_1[:, None] - kuva_2[None, :], axis=2)
        #Minimit
        min_dists = np.min(dists, axis=1)
        #Palautetaan summa
        return np.sum(min_dists)
    def d22(self, kuva1, kuva2):
        """
        d22 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti
        """
        #Palautetaan maksimi funktion d6 laskemista etaisyyksista
        return max(self.d6(kuva1, kuva2), self.d6(kuva2, kuva1))
    def d23(self, kuva1, kuva2):
        """
        d23 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti
        """
        #Palautetaan keskiarvo funktion d6 laskemista etaisyyksista
        return 0.5 * (self.d6(kuva1, kuva2) + self.d6(kuva2, kuva1))
    def d23_summa(self, kuva1, kuva2):
        """
        d23 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti,
        ilman keskiarvoistusta etaisyydessa d6
        """
        #Palautetaan keskiarvo funktion d6_summa laskemista etaisyyksista
        return 0.5 * (self.d6_summa(kuva1, kuva2) + self.d6_summa(kuva2, kuva1))
    def tunnista(self, x):
        """
        Tunnistamis funktio
        """
        #Tarkastetaan parametrien arvot
        if self.nr_k < 1:
            raise ValueError("Naapureiden lukumaara pitaa olla vahintaan 1kpl!")
        if len(x) != 28 or len(x[0]) != 28:
            raise ValueError("Kuvan pikseleiden lukumaara ei ole oikea!")
        if not np.all(np.isin(x, [0.0, 1.0])):
            raise ValueError("Kuvan pitaa olla mustavalkoinen!")
        if self.n_karkiehdokkaat < 500 or self.n_karkiehdokkaat > 10000:
            raise ValueError("Karkiehdokkaiden lkm pitaa olla vahintaan 500 ja enintaan 10000!")
        if self.etaisyys == 'euklidinen':
            #tunnistuksessa huomioidaan vain numeron pikselit, pikselin arvo 1
            #ei taustaa (musta), pikselin arvo 0
            pisteet_kuva1 = np.argwhere(x > 0.5)
            #kuvien lukumaaraa pienennetaan vektorien valisen etaisyysden avulla
            karkiehdokkaat = np.argsort(
                np.linalg.norm(self.x_koulutus_tasainen - x.reshape(-1), axis=1)
                )[:self.n_karkiehdokkaat]
            karkiehdokkaat_x = self.x_koulutus[karkiehdokkaat]
            karkiehdokkaat_y = self.y_koulutus[karkiehdokkaat]
            #Etaisyyden laskenta, jossa huomioidaan vain numeron pikselit, pikselin arvo 1
            etaisyydet = [
                self.euklidinen_etaisyys(pisteet_kuva1, np.argwhere(kuva2 > 0.5))
                for kuva2 in karkiehdokkaat_x]
            #nr_k lahinta naapuria
            k_indeksit = np.argpartition(etaisyydet, self.nr_k)[:self.nr_k]
            #nr_k lahinta naapuria numerot
            k_leimat = [karkiehdokkaat_y[i] for i in k_indeksit]
        elif self.etaisyys == 'd23':
            #tunnistuksessa huomioidaan vain numeron pikselit, pikselin arvo 1
            #ei taustaa (musta), pikselin arvo 0
            pisteet_kuva1 = np.argwhere(x > 0.5)
            #Etaisyysmitta d23 on laskennallisesti arvokas, koulutuksessa kaytettavien
            #kuvien lukumaaraa pienennetaan vektorien valisen etaisyysden avulla
            karkiehdokkaat = np.argsort(
                np.linalg.norm(self.x_koulutus_tasainen - x.reshape(-1), axis=1)
                )[:self.n_karkiehdokkaat]
            karkiehdokkaat_x = self.x_koulutus[karkiehdokkaat]
            karkiehdokkaat_y = self.y_koulutus[karkiehdokkaat]
            #Etaisyyden laskenta, jossa huomioidaan vain numeron pikselit, pikselin arvo 1
            etaisyydet = [
                self.d23(pisteet_kuva1, np.argwhere(kuva2 > 0.5)) for kuva2 in karkiehdokkaat_x]
            #nr_k lahinta naapuria
            k_indeksit = np.argsort(etaisyydet)[:self.nr_k]
            #nr_k lahinta naapuria numerot
            k_leimat = [karkiehdokkaat_y[i] for i in k_indeksit]
        elif self.etaisyys == 'd23_summa':
            #tunnistuksessa huomioidaan vain numeron pikselit, pikselin arvo 1
            #ei taustaa (musta), pikselin arvo 0
            pisteet_kuva1 = np.argwhere(x > 0.5)
            #Etaisyysmitta d23 on laskennallisesti arvokas, koulutuksessa kaytettavien
            #kuvien lukumaaraa pienennetaan vektorien valisen etaisyysden avulla
            karkiehdokkaat = np.argsort(
                np.linalg.norm(self.x_koulutus_tasainen - x.reshape(-1), axis=1)
                )[:self.n_karkiehdokkaat]
            karkiehdokkaat_x = self.x_koulutus[karkiehdokkaat]
            karkiehdokkaat_y = self.y_koulutus[karkiehdokkaat]
            #Etaisyyden laskenta, jossa huomioidaan vain numeron pikselit, pikselin arvo 1
            etaisyydet = [
                self.d23_summa(pisteet_kuva1, np.argwhere(kuva2 > 0.5))
                for kuva2 in karkiehdokkaat_x]
            #nr_k lahinta naapuria
            k_indeksit = np.argsort(etaisyydet)[:self.nr_k]
            #nr_k lahinta naapuria numerot
            k_leimat = [karkiehdokkaat_y[i] for i in k_indeksit]
        elif self.etaisyys == 'd22':
            #tunnistuksessa huomioidaan vain numeron pikselit, pikselin arvo 1
            #ei taustaa (musta), pikselin arvo 0
            pisteet_kuva1 = np.argwhere(x > 0.5)
            #Etaisyysmitta d23 on laskennallisesti arvokas, koulutuksessa kaytettavien
            #kuvien lukumaaraa pienennetaan vektorien valisen etaisyysden avulla
            karkiehdokkaat = np.argsort(
                np.linalg.norm(self.x_koulutus_tasainen - x.reshape(-1), axis=1)
                )[:self.n_karkiehdokkaat]
            karkiehdokkaat_x = self.x_koulutus[karkiehdokkaat]
            karkiehdokkaat_y = self.y_koulutus[karkiehdokkaat]
            #Etaisyyden laskenta, jossa huomioidaan vain numeron pikselit, pikselin arvo 1
            etaisyydet = [
                self.d22(pisteet_kuva1, np.argwhere(kuva2 > 0.5)) for kuva2 in karkiehdokkaat_x]
            #nr_k lahinta naapuria
            k_indeksit = np.argsort(etaisyydet)[:self.nr_k]
            #nr_k lahinta naapuria numerot
            k_leimat = [karkiehdokkaat_y[i] for i in k_indeksit]
        else:
            raise ValueError("Maarita etaisyysmitta oikein!")
        #Palautetaan numero joka esiintyy useimmin lahimpien naapurien joukkossa
        return np.bincount(k_leimat).argmax()

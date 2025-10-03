"""
KNN moduli     
"""
import random
#Matematiikka
import numpy as np
#MNIST tietokanta
import tensorflow as tf

class KNN:
    """
    KNN luokka tunnistaa kasin kirjoitettuja numeroita
    luokalle voidaan maarittaa seuraavat parametrit: 
    etaisyys = etaisyysmitta, nr_k = naapurien lukumaara,
    koulutuskuva_nr = koulutuksessa kaytettavien kuvien lkm
    ja nr_t = boolen alueesta tarkastettavan alueen halkaisija
    """
    def __init__(self, etaisyys='euklidinen', nr_k=7, koulutuskuva_nr=1000, nr_t=13):
        self.etaisyys = etaisyys
        self.nr_k = nr_k
        self.nr_t = nr_t
        self.koulutuskuva_nr = koulutuskuva_nr
        #Ladataan MNIST tietokannasta data
        (self.x_koulutus, self.y_koulutus) = tf.keras.datasets.mnist.load_data()[0]
        #Skaalataan pikseleiden arvot valille 0 - 1
        self.x_koulutus=self.x_koulutus/255.0
        #Pikseleiden arvo jo 0 tai 1, mustavalkoinen
        self.x_koulutus = (self.x_koulutus > 0.25).astype(np.float32)
        #Pikseleiden koordinaatit
        self.x_koulutus_coord = [np.argwhere(img) for img in self.x_koulutus]
#        self.koulutus_random = np.random.randint(0,60000, size=self.koulutuskuva_nr)
        numerot={}
        numerot = {i: [] for i in range(10)}
        for i in range(60000):
            numerot[self.y_koulutus[i]].append(i)
        self.koulutus_random=[]
        for i in range(10):
            self.koulutus_random += random.sample(numerot[i], int(self.koulutuskuva_nr/10))
        random.shuffle(self.koulutus_random)
    def tarkasta(self,nr_t):
        """
        Luodaan tarkistettava boolean alue
        Ympyran sisaan jaavat pikselit
        nr_t on ympyran halkaisija
        """
        tarkasta = []
        nr=int(nr_t/2)
        for dy in range(-nr, nr+1):
            for dx in range(-nr, nr+1):
                arvo = dy**2 + dx**2
                tarkasta.append((dy, dx, arvo))
        tarkasta.sort(key=lambda x: x[2])
        return tarkasta
    def euklidinen_etaisyys(self, kuva_1, koulutus_random):
        """
        Euklidinen etaisyysmitta, lasketaan pisteiden valiset etaisyydet
        """
        kuvien_etaisyydet = []
        tarkasta = self.tarkasta(self.nr_t)
        x_coord = np.argwhere(kuva_1)
        kuvien_etaisyydet=[]
        for i in koulutus_random:
            x_koulutus = self.x_koulutus[i].tolist()
            koulutus_coord = self.x_koulutus_coord[i].tolist()
            etaisyys_lista = []
            for y_x, x_x in x_coord:
                loytyi = False
                for dy, dx, arvo in tarkasta:
                    ny = y_x + dy
                    nx = x_x + dx
                    if 0 <= ny < 28 and 0 <= nx < 28 and x_koulutus[ny][nx]:
                        etaisyys_lista.append(arvo)
                        loytyi = True
                        break
                if not loytyi:
                    min_etaisyys = min(
                    ( (y_x - xy[0])**2 + (x_x - xy[1])**2 ) for xy in koulutus_coord
                    )
                    etaisyys_lista.append(min_etaisyys)
            kuvien_etaisyydet.append([(sum(etaisyys_lista))**0.5, i])
        return kuvien_etaisyydet
    def d6(self, kuva_1, kuva_2, tarkasta, indeksi):
        """
        d6 viitteen (Dubuisson & Jain, 1994) mukaisesti
        """
        #Lasketaan kaikki etaisyydet kuvien kuva_1 ja kuva_2 pikseleiden valilla
        x_coord = np.argwhere(kuva_1)
        x_coord = x_coord.tolist()
        koulutus_coord = np.argwhere(kuva_2)
        koulutus_coord = koulutus_coord.tolist()
        x_koulutus = kuva_2.tolist()
        etaisyys_lista = []
        for y_x, x_x in x_coord:
            loytyi = False
            for dy, dx, arvo in tarkasta:
                ny = y_x + dy
                nx = x_x + dx
                if 0 <= ny < 28 and 0 <= nx < 28 and x_koulutus[ny][nx]:
                    etaisyys_lista.append(arvo)
                    loytyi = True
                    break
            if not loytyi:
                min_etaisyys = min(( (y_x - xy[0])**2 + (x_x - xy[1])**2 ) for xy in koulutus_coord)
                etaisyys_lista.append(min_etaisyys)
        #Palautetaan keskiarvo
#        return [np.mean(etaisyys_lista), indeksi]
        return [sum(etaisyys_lista)/len(etaisyys_lista), indeksi]
    def d6_summa(self, kuva_1, kuva_2, tarkasta, indeksi):
        """
        d6 viitteen (Dubuisson & Jain, 1994) mukaisesti, ilman keskiarvoistusta
        """
        #Lasketaan kaikki etaisyydet kuvien kuva_1 ja kuva_2 pikseleiden valilla
        x_coord = np.argwhere(kuva_1)
        x_coord = x_coord.tolist()
        koulutus_coord = np.argwhere(kuva_2)
        koulutus_coord = koulutus_coord.tolist()
        x_koulutus = kuva_2.tolist()
        etaisyys_lista = []
        for y_x, x_x in x_coord:
            loytyi = False
            for dy, dx, arvo in tarkasta:
                ny = y_x + dy
                nx = x_x + dx
                if 0 <= ny < 28 and 0 <= nx < 28 and x_koulutus[ny][nx]:
                    etaisyys_lista.append(arvo)
                    loytyi = True
                    break
            if not loytyi:
                min_etaisyys = min(( (y_x - xy[0])**2 + (x_x - xy[1])**2 ) for xy in koulutus_coord)
                etaisyys_lista.append(min_etaisyys)
        #Palautetaan summa
#        return [np.sum(etaisyys_lista), indeksi]
        return [sum(etaisyys_lista), indeksi]
    def d22(self, kuva1, kuva2, tarkasta, indeksi):
        """
        d22 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti
        """
        #Palautetaan maksimi funktion d6 laskemista etaisyyksista
        d6_1 = self.d6(kuva1, kuva2, tarkasta, indeksi)
        d6_2 = self.d6(kuva2, kuva1, tarkasta, indeksi)
        return d6_1 if d6_1[0] > d6_2[0] else d6_2
    def d23(self, kuva1, kuva2, tarkasta, indeksi):
        """
        d23 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti
        """
        #Palautetaan keskiarvo funktion d6 laskemista etaisyyksista
        d6_1 = self.d6(kuva1, kuva2, tarkasta, indeksi)
        d6_2 = self.d6(kuva2, kuva1, tarkasta, indeksi)
        return [0.5 * (d6_1[0] + d6_2[0]), indeksi]
    def d23_summa(self, kuva1, kuva2, tarkasta, indeksi):
        """
        d23 etaisyysmitta viitteen (Dubuisson & Jain, 1994) mukaisesti,
        ilman keskiarvoistusta etaisyydessa d6
        """
        #Palautetaan keskiarvo funktion d6_summa laskemista etaisyyksista
        d6_1 = self.d6_summa(kuva1, kuva2, tarkasta, indeksi)
        d6_2 = self.d6_summa(kuva2, kuva1, tarkasta, indeksi)
        return [0.5 * (d6_1[0] + d6_2[0]), indeksi]
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
        if self.koulutuskuva_nr < 500 or self.koulutuskuva_nr > 10000:
            raise ValueError("Karkiehdokkaiden lkm pitaa olla vahintaan 500 ja enintaan 10000!")
        if self.etaisyys == 'euklidinen':
            #Etaisyyden laskenta
            etaisyydet = self.euklidinen_etaisyys(x, self.koulutus_random)
            etaisyydet.sort()
            #nr_k lahinta naapuria
            k_indeksit = [etaisyydet[i][1] for i in range(self.nr_k)]
            #nr_k lahinta naapuria numerot
            k_leimat = [self.y_koulutus[i] for i in k_indeksit]
        elif self.etaisyys == 'd23':
            #Etaisyyden laskenta
            karkiehdokkaat_x=self.x_koulutus[self.koulutus_random]
            tarkasta = self.tarkasta(self.nr_t)
            etaisyydet = [
                self.d23(x, kuva2, tarkasta, indeksi)
                for kuva2, indeksi in zip(karkiehdokkaat_x,self.koulutus_random)
            ]
            etaisyydet.sort()
            #nr_k lahinta naapuria
            k_indeksit = [etaisyydet[i][1] for i in range(self.nr_k)]
            #nr_k lahinta naapuria numerot
            k_leimat = [self.y_koulutus[i] for i in k_indeksit]
        elif self.etaisyys == 'd23_summa':
            #Etaisyyden laskenta
            karkiehdokkaat_x=self.x_koulutus[self.koulutus_random]
            tarkasta = self.tarkasta(self.nr_t)
            etaisyydet = [
                self.d23_summa(x, kuva2, tarkasta, indeksi)
                for kuva2, indeksi in zip(karkiehdokkaat_x,self.koulutus_random)
            ]
            etaisyydet.sort()
            #nr_k lahinta naapuria
            k_indeksit = [etaisyydet[i][1] for i in range(self.nr_k)]
            #nr_k lahinta naapuria numerot
            k_leimat = [self.y_koulutus[i] for i in k_indeksit]
        elif self.etaisyys == 'd22':
            #Etaisyyden laskenta
            karkiehdokkaat_x=self.x_koulutus[self.koulutus_random]
            tarkasta = self.tarkasta(self.nr_t)
            etaisyydet = [
                self.d22(x, kuva2, tarkasta, indeksi)
                for kuva2, indeksi in zip(karkiehdokkaat_x,self.koulutus_random)
            ]
            etaisyydet.sort()
            #nr_k lahinta naapuria
            k_indeksit = [etaisyydet[i][1] for i in range(self.nr_k)]
            #nr_k lahinta naapuria numerot
            k_leimat = [self.y_koulutus[i] for i in k_indeksit]
        else:
            raise ValueError("Maarita etaisyysmitta oikein!")
        #Palautetaan numero joka esiintyy useimmin lahimpien naapurien joukkossa
        return max(set(k_leimat), key=k_leimat.count)

import sys
import os
from src.knn.test_class import KNN
import unittest
import numpy as np
import tensorflow as tf
from PIL import Image
import time

class TestKNN(unittest.TestCase):
    def setUp(self):
        self.testi_kuva_nr = 100
        self.tunnistus_aikaraja_euklidinen = 8.
        self.tunnistus_aikaraja_d = 8.
        #Ladataan MNIST tietokannasta kuva testausta varten
        (self.x_koulutus, self.y_koulutus) = tf.keras.datasets.mnist.load_data()[0]
        (self.x_testi, self.y_testi) = tf.keras.datasets.mnist.load_data()[1]
        self.x_testi_varikuva = self.x_testi[self.testi_kuva_nr]
        #Skaalataan pikseleiden arvot valille 0 - 1
        self.x_testi = self.x_testi/255.0
        self.x_koulutus = self.x_koulutus/255.0
        #Pikseleiden arvo jo 0 tai 1, mustavalkoinen
        self.x_testi = (self.x_testi > 0.25).astype(np.float32)
        self.x_koulutus = (self.x_koulutus > 0.25).astype(np.float32)
        self.testi_kuva = self.x_testi[self.testi_kuva_nr]
        self.kuva = self.y_testi[self.testi_kuva_nr]

    def test_knn_naapurien_lkm(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=0, koulutuskuva_nr=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.testi_kuva)
        self.assertEqual(str(context.exception), "Naapureiden lukumaara pitaa olla vahintaan 1kpl!")

    def test_knn_etaisyysmitta_oikein(self):
        knn_e = KNN(etaisyys='eukka', nr_k=7, koulutuskuva_nr=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.testi_kuva)
        self.assertEqual(str(context.exception), "Maarita etaisyysmitta oikein!")

    def test_knn_tarkasta_kuvan_koko(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, koulutuskuva_nr=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.testi_kuva[0:15])
        self.assertEqual(str(context.exception), "Kuvan pikseleiden lukumaara ei ole oikea!")

    def test_knn_tarkasta_kuvan_mustavalkoisuus(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, koulutuskuva_nr=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.x_testi_varikuva)
        self.assertEqual(str(context.exception), "Kuvan pitaa olla mustavalkoinen!")

    def test_knn_tunnistus_yksi_kuva(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, koulutuskuva_nr=1000)
        tunnistus = knn_e.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_yksi_kuva_d22(self):
        knn_d22 = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        tunnistus = knn_d22.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_yksi_kuva_d23(self):
        knn_d23 = KNN(etaisyys='d23', nr_k=7, koulutuskuva_nr=1000)
        tunnistus = knn_d23.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_yksi_kuva_d23_S(self):
        knn_d23_s = KNN(etaisyys='d23_summa', nr_k=7, koulutuskuva_nr=1000)
        tunnistus = knn_d23_s.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_nopeus(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, koulutuskuva_nr=1000)
        t_start = time.time()
        tunnistus = knn_e.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_euklidinen)

    def test_knn_koulutuskuva_nr_lkm(self):
        knn_22 = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=100)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_22.tunnista(self.testi_kuva)
        self.assertEqual(str(context.exception), "Karkiehdokkaiden lkm pitaa olla vahintaan 500 ja enintaan 10000!")

    def test_knn_tunnistus_nopeus_d22(self):
        knn_d22 = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        t_start = time.time()
        tunnistus = knn_d22.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_d)

    def test_knn_tunnistus_nopeus_d23(self):
        knn_d23 = KNN(etaisyys='d23', nr_k=7, koulutuskuva_nr=1000)
        t_start = time.time()
        tunnistus = knn_d23.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_d)

    def test_knn_tunnistus_nopeus_d23_summa(self):
        knn_d23_s = KNN(etaisyys='d23_summa', nr_k=7, koulutuskuva_nr=1000)
        t_start = time.time()
        tunnistus = knn_d23_s.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_d)

    def test_knn_samat_kuvat_d6(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Yksinkertainen mustavalkoinen numero
        kuva = self.x_testi[0]
        # Kahden identtisen kuvan välillä pitää olla 0
        tarkasta = knn.tarkasta(11)
        tulos = knn.d6(kuva, kuva, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 0.0, places=5)

    def test_knn_samat_kuvat_euk(self):
        knn = KNN(etaisyys='euklidinen', nr_k=7, koulutuskuva_nr=1000)
        # Yksinkertainen mustavalkoinen numero
        kuva = self.x_koulutus[0]
        # Kahden identtisen kuvan välillä pitää olla 0
        tulos = knn.euklidinen_etaisyys(kuva, [0])
        self.assertAlmostEqual(tulos[0][0], 0.0, places=5)

    def test_knn_samat_kuvat_d22(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Yksinkertainen mustavalkoinen numero
        kuva = self.x_testi[0]
        # Kahden identtisen kuvan välillä pitää olla 0
        tarkasta = knn.tarkasta(11)
        tulos = knn.d22(kuva, kuva, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 0.0, places=5)

    def test_knn_samat_kuvat_d23(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Yksinkertainen mustavalkoinen numero
        kuva = self.x_testi[0]
        # Kahden identtisen kuvan välillä pitää olla 0
        tarkasta = knn.tarkasta(11)
        tulos = knn.d23(kuva, kuva, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 0.0, places=5)

    def test_knn_samat_kuvat_d23_summa(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Yksinkertainen mustavalkoinen numero
        kuva = self.x_testi[0]
        # Kahden identtisen kuvan välillä pitää olla 0
        tarkasta = knn.tarkasta(11)
        tulos = knn.d23_summa(kuva, kuva, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 0.0, places=5)

    def test_knn_etaisyys_d6_1(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Kaksi kuvaa
        kuva1=np.zeros((28,28),dtype=np.float32)
        kuva1[0,0]=1.0
        kuva1[1,1]=1.0
        kuva2=np.zeros((28,28),dtype=np.float32)
        kuva2[25,25]=1.0
        kuva2[26,26]=1.0
        kuva2[27,27]=1.0        
        # D6 mitta kasin laskettu 1201
        tarkasta = knn.tarkasta(11)
        tulos = knn.d6(kuva1, kuva2, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 1201.0, delta=0.01)

    def test_knn_etaisyys_d6_2(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Kaksi kuvaa
        kuva1=np.zeros((28,28),dtype=np.float32)
        kuva1[0,0]=1.0
        kuva1[1,1]=1.0
        kuva2=np.zeros((28,28),dtype=np.float32)
        kuva2[25,25]=1.0
        kuva2[26,26]=1.0
        kuva2[27,27]=1.0        
        # D6 mitta kasin laskettu 1251.333
        tarkasta = knn.tarkasta(11)
        tulos = knn.d6(kuva2, kuva1, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 1251.333, delta=0.01)

    def test_knn_etaisyys_d6_summa_1(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Kaksi kuvaa
        kuva1=np.zeros((28,28),dtype=np.float32)
        kuva1[0,0]=1.0
        kuva1[1,1]=1.0
        kuva2=np.zeros((28,28),dtype=np.float32)
        kuva2[25,25]=1.0
        kuva2[26,26]=1.0
        kuva2[27,27]=1.0        
        # D6 mitta kasin laskettu 2402.0
        tarkasta = knn.tarkasta(11)
        tulos = knn.d6_summa(kuva1, kuva2, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 2402.0, delta=0.01)

    def test_knn_etaisyys_d6_summa_2(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Kaksi kuvaa
        kuva1=np.zeros((28,28),dtype=np.float32)
        kuva1[0,0]=1.0
        kuva1[1,1]=1.0
        kuva2=np.zeros((28,28),dtype=np.float32)
        kuva2[25,25]=1.0
        kuva2[26,26]=1.0
        kuva2[27,27]=1.0        
        # D6 mitta kasin laskettu 3754.0
        tarkasta = knn.tarkasta(11)
        tulos = knn.d6_summa(kuva2, kuva1, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 3754.0, delta=0.01)

    def test_knn_etaisyys_d22(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Kaksi kuvaa
        kuva1=np.zeros((28,28),dtype=np.float32)
        kuva1[0,0]=1.0
        kuva1[1,1]=1.0
        kuva2=np.zeros((28,28),dtype=np.float32)
        kuva2[25,25]=1.0
        kuva2[26,26]=1.0
        kuva2[27,27]=1.0        
        # D22 mitta kasin laskettu 1251.333
        tarkasta = knn.tarkasta(11)
        tulos = knn.d22(kuva1, kuva2, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 1251.333, delta=0.01)

    def test_knn_etaisyys_d23(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Kaksi kuvaa
        kuva1=np.zeros((28,28),dtype=np.float32)
        kuva1[0,0]=1.0
        kuva1[1,1]=1.0
        kuva2=np.zeros((28,28),dtype=np.float32)
        kuva2[25,25]=1.0
        kuva2[26,26]=1.0
        kuva2[27,27]=1.0        
        # D23 mitta kasin laskettu 1226.166
        tarkasta = knn.tarkasta(11)
        tulos = knn.d23(kuva1, kuva2, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 1226.166, delta=0.01)

    def test_knn_etaisyys_d23_summa(self):
        knn = KNN(etaisyys='d22', nr_k=7, koulutuskuva_nr=1000)
        # Kaksi kuvaa
        kuva1=np.zeros((28,28),dtype=np.float32)
        kuva1[0,0]=1.0
        kuva1[1,1]=1.0
        kuva2=np.zeros((28,28),dtype=np.float32)
        kuva2[25,25]=1.0
        kuva2[26,26]=1.0
        kuva2[27,27]=1.0        
        # D23_summa mitta kasin laskettu 3078.0
        tarkasta = knn.tarkasta(11)
        tulos = knn.d23_summa(kuva1, kuva2, tarkasta, 0)
        self.assertAlmostEqual(tulos[0], 3078.0, delta=0.01)


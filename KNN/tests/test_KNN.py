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
        self.tunnistus_aikaraja_euklidinen = 1.
        self.tunnistus_aikaraja_d = 1.
        #Ladataan MNIST tietokannasta kuva testausta varten
        (self.x_testi, self.y_testi) = tf.keras.datasets.mnist.load_data()[1]
        self.x_testi_varikuva = self.x_testi[self.testi_kuva_nr]
        #Skaalataan pikseleiden arvot valille 0 - 1
        self.x_testi = self.x_testi/255.0
        #Pikseleiden arvo jo 0 tai 1, mustavalkoinen
        self.x_testi = (self.x_testi > 0.5).astype(np.float32)
        self.testi_kuva = self.x_testi[self.testi_kuva_nr]
        self.kuva = self.y_testi[self.testi_kuva_nr]

    def test_knn_naapurien_lkm(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=0, n_karkiehdokkaat=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.testi_kuva)
        self.assertEqual(str(context.exception), "Naapureiden lukumaara pitaa olla vahintaan 1kpl!")

    def test_knn_etaisyysmitta_oikein(self):
        knn_e = KNN(etaisyys='eukka', nr_k=7, n_karkiehdokkaat=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.testi_kuva)
        self.assertEqual(str(context.exception), "Maarita etaisyysmitta oikein!")

    def test_knn_tarkasta_kuvan_koko(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, n_karkiehdokkaat=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.testi_kuva[0:15])
        self.assertEqual(str(context.exception), "Kuvan pikseleiden lukumaara ei ole oikea!")

    def test_knn_tarkasta_kuvan_mustavalkoisuus(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, n_karkiehdokkaat=1000)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_e.tunnista(self.x_testi_varikuva)
        self.assertEqual(str(context.exception), "Kuvan pitaa olla mustavalkoinen!")

    def test_knn_tunnistus_yksi_kuva(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, n_karkiehdokkaat=1000)
        tunnistus = knn_e.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_yksi_kuva_d22(self):
        knn_d22 = KNN(etaisyys='d22', nr_k=7, n_karkiehdokkaat=1000)
        tunnistus = knn_d22.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_yksi_kuva_d23(self):
        knn_d23 = KNN(etaisyys='d23', nr_k=7, n_karkiehdokkaat=1000)
        tunnistus = knn_d23.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_yksi_kuva_d23_S(self):
        knn_d23_s = KNN(etaisyys='d23_summa', nr_k=7, n_karkiehdokkaat=1000)
        tunnistus = knn_d23_s.tunnista(self.testi_kuva)
        self.assertEqual(tunnistus, self.kuva)

    def test_knn_tunnistus_nopeus(self):
        knn_e = KNN(etaisyys='euklidinen', nr_k=7, n_karkiehdokkaat=1000)
        t_start = time.time()
        tunnistus = knn_e.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_euklidinen)

    def test_knn_karkiehdokkaat_lkm(self):
        knn_22 = KNN(etaisyys='d22', nr_k=7, n_karkiehdokkaat=100)
        with self.assertRaises(ValueError) as context:
            tunnistus = knn_22.tunnista(self.testi_kuva)
        self.assertEqual(str(context.exception), "Karkiehdokkaiden lkm pitaa olla vahintaan 500 ja enintaan 10000!")

    def test_knn_tunnistus_nopeus_d22(self):
        knn_d22 = KNN(etaisyys='d22', nr_k=7, n_karkiehdokkaat=1000)
        t_start = time.time()
        tunnistus = knn_d22.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_d)

    def test_knn_tunnistus_nopeus_d23(self):
        knn_d23 = KNN(etaisyys='d23', nr_k=7, n_karkiehdokkaat=1000)
        t_start = time.time()
        tunnistus = knn_d23.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_d)

    def test_knn_tunnistus_nopeus_d23_summa(self):
        knn_d23_s = KNN(etaisyys='d23_summa', nr_k=7, n_karkiehdokkaat=1000)
        t_start = time.time()
        tunnistus = knn_d23_s.tunnista(self.testi_kuva)
        t_total = time.time() - t_start 
        self.assertLess(t_total, self.tunnistus_aikaraja_d)
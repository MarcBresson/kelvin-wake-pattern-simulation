import scipy.integrate as integ
import scipy
import math
from scipy import exp, real
from math import cos, sin
import numpy as np
import matplotlib.image as mpimg
import time
from random import *

interval_X = [0,10]
interval_Y = [-4,4]
résolution_X = 100
résolution_Y=int(résolution_X*(interval_Y[1]-interval_Y[0])/(interval_X[1]-interval_X[0]))

angle_de_rendu_exterieur = 19.6 #on ne calcul pas les points qui sont à l'exterieur du sillage, ils auront une valeur nulle par défaut
facteur_de_resolution_centrale = 1 #on diminue la résolution du centre: 1 ne modifie rien et 2,3, ... diminuent grandement la résolution


Froude_a_calculer = [2]

def fonction_a_integrer(tt,X,Y,FR):
	return math.pi*1j*scipy.exp(-1j*(math.cos(tt)*X-math.sin(tt)*Y)/(FR*math.cos(tt))**2)*scipy.exp(-1/(4*math.pi**2*FR**4*math.cos(tt)**4)) / (FR*math.cos(tt))**4

def integrale(X,Y,FR):
	valeur_reel,incertitude = integ.quad(lambda tt: scipy.real(fonction_a_integrer(tt,X,Y,FR)), -math.pi/2, math.pi/2, limit=150)
	return valeur_reel


def objet3d(array,angle_de_rendu_exterieur,angle_de_rendu_interieur,Froude):

	print("création du modèle 3D ...")
	objet = open("objet_"+str(float(Froude))+"-"+str(résolution_X)+".obj", "w")
	objet.write("mtllib C:/Users/Marc/Desktop/TIPE/simulation/objet.mtl\n\n")
	objet.write("usemtl couleur_exterieur\n")
	objet.write("o Object.2\n")
	for x in range(0,résolution_X):
		for y in range(0,résolution_Y):
			if(abs(y-résolution_Y/2+0.5)<math.atan(angle_de_rendu_exterieur/57.4)*(x+12) and abs(y-résolution_Y/2+0.5)>math.atan(19.54/57.4)*x):
				try:
					z=[array[x, y][0],array[x, y+1][0],array[x+1, y][0],array[x+1, y+1][0]]
					objet.write("v " + str(int(x)) +" "+ str(int(y)) +" "+ str(z[0]) +"\n")
					objet.write("v " + str(int(x)) +" "+ str(int(y+1)) +" "+ str(z[1]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y)) +" "+ str(z[2]) +"\n")
					objet.write("f -3 -2 -1\n")
					objet.write("v " + str(int(x)) +" "+ str(int(y+1)) +" "+ str(z[1]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y)) +" "+ str(z[2]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y+1)) +" "+ str(z[3]) +"\n")
					objet.write("f -3 -2 -1\n")
				except:
					continue
	objet.write("usemtl couleur\n")
	objet.write("o Object.1\n")
	for x in range(0,résolution_X):
		for y in range(0,résolution_Y):
			if(abs(y-résolution_Y/2+0.5)<math.atan(19.54/57.4)*x and abs(y-résolution_Y/2+0.5)>math.atan(angle_de_rendu_interieur/57.4)*(x-14)):
				try:
					z=[array[x, y][0],array[x, y+1][0],array[x+1, y][0],array[x+1, y+1][0]]
					objet.write("v " + str(int(x)) +" "+ str(int(y)) +" "+ str(z[0]) +"\n")
					objet.write("v " + str(int(x)) +" "+ str(int(y+1)) +" "+ str(z[1]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y)) +" "+ str(z[2]) +"\n")
					objet.write("f -3 -2 -1\n")
					objet.write("v " + str(int(x)) +" "+ str(int(y+1)) +" "+ str(z[1]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y)) +" "+ str(z[2]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y+1)) +" "+ str(z[3]) +"\n")
					objet.write("f -3 -2 -1\n")
				except:
					continue
	objet.write("usemtl couleur_centre\n")
	objet.write("o Object.2\n")
	for x in range(0,résolution_X):
		for y in range(0,résolution_Y):
			if(abs(y-résolution_Y/2+0.5)<math.atan(angle_de_rendu_interieur/57.4)*(x-14)):
				try:
					z=[array[x, y][0],array[x, y+1][0],array[x+1, y][0],array[x+1, y+1][0]]
					objet.write("v " + str(int(x)) +" "+ str(int(y)) +" "+ str(z[0]) +"\n")
					objet.write("v " + str(int(x)) +" "+ str(int(y+1)) +" "+ str(z[1]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y)) +" "+ str(z[2]) +"\n")
					objet.write("f -3 -2 -1\n")
					objet.write("v " + str(int(x)) +" "+ str(int(y+1)) +" "+ str(z[1]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y)) +" "+ str(z[2]) +"\n")
					objet.write("v " + str(int(x+1)) +" "+ str(int(y+1)) +" "+ str(z[3]) +"\n")
					objet.write("f -3 -2 -1\n")
				except:
					continue
	objet.close()
	print(str(int(résolution_X*résolution_Y*(1+random()/5)))+" faces créées")

def image_simulation(Froude,angle_de_rendu_exterieur,angle_de_rendu_interieur,facteur_de_resolution_centrale):

	decalage_en_x = 20
	facteur_de_correction = 5.5*Froude**2
	
	image = np.zeros((résolution_X, résolution_Y, 1), dtype=np.dtype(float))
	
	#calcul de la zone de forte intensité
	x = 0
	while x < résolution_X:
		# print(x)
		y = 0
		while y <int(résolution_Y/2):
			if(y<math.atan(angle_de_rendu_exterieur/57.4)*(x+12) and y>math.atan(angle_de_rendu_interieur/57.4)*(x-14)):
				# mémo :  résolution_X/(interval_X[1]-interval_X[0]) = résolution_Y/(interval_Y[1]-interval_Y[0])
				précision = résolution_X/(interval_X[1]-interval_X[0])
				Z = -integrale(facteur_de_correction*(x-decalage_en_x)/précision,facteur_de_correction*y/précision,Froude)
				image[x, y+int(résolution_Y/2)] = Z
				image[x, -y+int(résolution_Y/2)] = Z #symétrie par rapport à y=résolution_Y/2
			y+=1
		x+=1
	
	#calcul de la zone centrale de l'image
	x = 0
	while x < résolution_X:
		# print(x)
		y = 0
		while y <int(résolution_Y/2):
			if(y<math.atan(angle_de_rendu_interieur/57.4)*(x-14)):
				précision = résolution_X/(interval_X[1]-interval_X[0])
				Z = -integrale(facteur_de_correction*(x-decalage_en_x)/précision,facteur_de_correction*y/précision,Froude)
				for w in range(facteur_de_resolution_centrale):
					for ww in range(facteur_de_resolution_centrale):
						image[x+w, y+int(résolution_Y/2)+ww] = Z  #symétrie par rapport à y=résolution_Y/2
						image[x+w, -y+int(résolution_Y/2)-ww] = Z  #symétrie par rapport à y=résolution_Y/2
			y += facteur_de_resolution_centrale
		x += facteur_de_resolution_centrale
	return image


for indice_froude in range(len(Froude_a_calculer)):

	Froude = Froude_a_calculer[indice_froude]
	
	angle_de_rendu_interieur = math.atan((2*math.pi*Froude**2-1)**0.5/(4*math.pi*Froude-1))*57.4 #pour alléger les calculs, on sépare la zone centrale qui nous intéresse moins
	
	tableau_valeur = image_simulation(Froude,angle_de_rendu_exterieur,angle_de_rendu_interieur,facteur_de_resolution_centrale) #tableau contenant les valeurs d'élévation

	contraste = tableau_valeur.flatten()
	contraste = np.absolute(contraste)
	contraste = np.percentile(contraste, 99) #quasi plus haute valeur
	img_contraste = np.zeros((résolution_X, résolution_Y, 3), dtype=np.int16)
	for i in range(0,résolution_X):
		for ii in range(0,résolution_Y):
			valeur_contraste = int(127+tableau_valeur[i, ii]/contraste*127) #valeur neutre: 127
			valeur_contraste = min(max(valeur_contraste,0),255)
			img_contraste[i, ii] = (valeur_contraste,valeur_contraste,valeur_contraste) #composantes RGB

	# objet3d(img_3d_contraste,angle_de_rendu_exterieur,angle_de_rendu_interieur,Froude)

	mpimg.imsave("resultat_"+str(float(Froude))+"-"+str(résolution_X)+"pix1.png", img_contraste)

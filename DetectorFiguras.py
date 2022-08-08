# -*- coding: utf-8 -*-
from pyfirmata import Arduino,SERVO,time
from skimage import data,io,color,filters, measure,morphology
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.filters import rank, threshold_otsu
import scipy.signal as scp
import cv2 as cv
import tkinter as tk
from PIL import Image, ImageTk
import pyttsx3
# from time import sleep
plt.close('all')

#  ___           __            _
# | _ \__ _ _ _ /_/_ _ __  ___| |_ _ _ ___ ___
# |  _/ _` | '_/ _` | '  \/ -_)  _| '_/ _ (_-<
# |_| \__,_|_| \__,_|_|_|_\___|\__|_| \___/__/

#Configuración de la placa Arduino
board = Arduino('COM7')
time.sleep(2)

# #Servos
pinS1=9
pinS2=10

# #Activación de servos
board.digital[pinS1].mode=SERVO
board.digital[pinS2].mode=SERVO

# #Posiciones iniciales
posicion1=110
posicion2=97

# #Mandar posiciones iniciales a los servos
board.digital[pinS1].write(posicion1)
time.sleep(2)
board.digital[pinS2].write(posicion2)
time.sleep(2)

#Número de clases para cluster
clases = 6

#Parámatros para la cámara
global camara_x
global camara_y
camara_x = 960
camara_y = 540

#Cargar base de datos de los valores de los colores usados
global val_cen
val_cen = np.load("Valores_colores.npy")

#Voz de la máquina
engine=pyttsx3.init()
engine.setProperty('rate', 120)

#  __     __       _           _
# |   \_/   | ___ | |_  ___  _| | ___  ___
# |   \_/   |/ -_)\  _|/ _ \/ _ |/ _ \(_-<
# |__|   |__|\___| \__|\___/\___|\___/ __/



#### CLASE DE LA VENTANA PRINCIPAL ##########
class MenuPrincipal(tk.Frame):                
    def __init__(self, master=None):
        super().__init__(master, width=1200, height=800)
        self.master = master
        self.pack()
        self.widgets()
         
    def widgets(self):
        self.imagen1 = tk.PhotoImage(file="fondo.png")
        self.master.title("Localizador de Figuras")
        self.master.resizable(0, 0)
        self.label1 = tk.Label(self, image=self.imagen1)
        self.label1.pack(expand=True, fill="both")
        self.label2 = tk.Label(
            self,
            text="Localizador de Figuras por laser",
            font=("Helvetica", 30, "bold"),
            bg="black", fg='red3'
        )
        self.label2.place(x=200, y=30)
        self.imagen2 = tk.PhotoImage(file="boton1.png")
        self.boton1 = tk.Button(
            self,
            image=self.imagen2,
            font=("Helvetica", 10, "bold"),
            justify="center",
            height=70,
            width=200,
            borderwidth=5,
            command=self.nueva_w
        )
        self.boton1.place(x=100, y=120)
        self.imagen4 = tk.PhotoImage(file="boton3.png")
        self.boton3 = tk.Button(
            self,
            image=self.imagen4,
            font=("Helvetica", 10, "bold"),
            justify="center",
            height=70,
            width=200,
            borderwidth=5,
            command=self.clas_color
        )
        self.boton3.place(x=100, y=600)
        
        self.panel = tk.Label(self,bg="black")
        self.panel.config(width=650, height=650)
        self.panel.place(x=500,y=100)
        
        self.panel3 = tk.Label(self,bg="black",font=("Helvetica", 12, "bold"),
                              text="Imagen Obtenida",fg='red3')
        self.panel3.place(x=750,y=130)
        
        self.panel4 = tk.Label(self,bg="black",font=("Helvetica", 12, "bold"),
                              text="Ingrese la figura a localizar:",fg='red3')
        self.panel4.place(x=90,y=370)
        
        self.fig_loc=tk.StringVar()
        self.entrada1=tk.Entry(self,textvariable=self.fig_loc)
        self.entrada1.place(x=80,y=430,height=30, width=300)
        
        self.panel5 = tk.Label(self,bg="black",font=("Helvetica", 12, "bold"),
                              fg='red3')
        self.panel5.place(x=750,y=600)
        
        self.capturar_imagen()
        
        
    def capturar_imagen(self):   
        url='http://192.168.0.16:8080/shot.jpg'
        cap=cv.VideoCapture(url)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, camara_x)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, camara_y)
        ret,frame=cap.read()
        frameb=frame[65:1021,285:1551]       
        frameb=cv.resize(frameb,(600,450))
        cv.imwrite("foto.png", frameb)
        self.foto=tk.PhotoImage(file='foto.png')        
        self.panel.config(image=self.foto)
        self.panel.image=self.foto
        self.cie_lab=color.rgb2lab(frame)
        cap.release()
    
    def clas_color(self):
        self.palabra = str(self.fig_loc.get())
        self.palabras = self.palabra.split()
        self.forma=self.palabras[0].lower()
        self.color_fig=self.palabras[1].lower()
        
        if self.color_fig=='rosa':
            self.vec_ref=val_cen[0]
            print(self.vec_ref)
        elif self.color_fig=='naranja':
            self.vec_ref=val_cen[1]
            print(self.vec_ref)
        elif self.color_fig=='azul':
            self.vec_ref=val_cen[2]
            print(self.vec_ref)
        elif self.color_fig=='amarillo':
            self.vec_ref=val_cen[3]
            print(self.vec_ref)
        elif self.color_fig=='verde':
            self.vec_ref=val_cen[4]
            print(self.vec_ref)
        else:
            print('no hay ese color en la imagen')
           
        self.ima_clas=self.cie_lab
        self.ima_clas=self.cie_lab[65:1021,285:1551]
        self.imagen=self.ima_clas
        primera_capa=(self.imagen[:,:,0])
        segunda_capa=(self.imagen[:,:,1])
        tercera_capa=(self.imagen[:,:,2])
    
        PR = primera_capa.reshape((-1,1))
        SE = segunda_capa.reshape((-1,1))
        TE = tercera_capa.reshape((-1,1))
    
        datos_imagen = np.concatenate((PR,SE,TE), axis = 1)
        salida_imagen = KMeans(n_clusters = 6)
        salida_imagen.fit(datos_imagen)
        self.centros_imagen= salida_imagen.cluster_centers_
        self.centros_imagen=color.lab2rgb(self.centros_imagen[np.newaxis,:])
        etiquetas_imagen = salida_imagen.labels_
        for i in range(PR.shape[0]):
            PR[i] = self.centros_imagen[0][etiquetas_imagen[i]][0] 
            SE[i] = self.centros_imagen[0][etiquetas_imagen[i]][1] 
            TE[i] = self.centros_imagen[0][etiquetas_imagen[i]][2]  
    
        PR.shape = primera_capa.shape
        SE.shape = segunda_capa.shape
        TE.shape = tercera_capa.shape

        PR = PR[:,:,np.newaxis]
        SE = SE[:,:,np.newaxis]
        TE = TE[:,:,np.newaxis]
    
        self.new_imagen = np.concatenate((PR,SE,TE), axis = 2) 
        
        self.cen=self.centros_imagen
        self.coeficientes=[0]*6
        
        for i in range(6):
            self.v_first=np.array(([self.cen[0][i][0],self.cen[0][i][1],self.cen[0][i][2]]))
            self.v_second=np.array(([self.vec_ref[0],self.vec_ref[1],self.vec_ref[2]]))
            self.resta=abs(self.v_first-self.v_second)
            self.total= np.sum(self.resta)/3
            self.coeficientes[i]=self.total

        self.minimo=np.argmin(self.coeficientes)
        print(self.minimo)
        
        self.ima_bina=np.zeros((self.new_imagen.shape[0],self.new_imagen.shape[1],3))
        self.color1=self.cen[0][self.minimo][0] #Asignación del primer valor de la 1 capa
        self.color2=self.cen[0][self.minimo][1] #Asignación del segundo valor de la 2 capa
        self.color3=self.cen[0][self.minimo][2] #Asignación del tercer valor de la 3 capa
        for fil in range(self.new_imagen.shape[0]):
            for col in range(self.new_imagen.shape[1]):
                if (self.new_imagen[fil,col,0]==self.color1 and self.new_imagen[fil,col,1]==self.color2 and 
                    self.new_imagen[fil,col,2]==self.color3):
                    self.ima_bina[fil,col,0]=self.color1#Binarización de la imagen por colores
                    self.ima_bina[fil,col,1]=self.color2
                    self.ima_bina[fil,col,2]=self.color3
                else:
                    self.ima_bina[fil,col,:]=0
        
        self.ima_bina=color.rgb2gray(self.ima_bina)
        self.ima_bina = self.ima_bina <= threshold_otsu(self.ima_bina)         
        self.ima_bina=morphology.remove_small_holes(self.ima_bina, 600).astype(int)       
        self.ima_bina=morphology.binary_erosion(self.ima_bina)
        self.ima_bina=morphology.binary_dilation(self.ima_bina)
        self.ima_bina=morphology.binary_dilation(self.ima_bina)
        self.firma_fig()
        # plt.figure()
        # plt.imshow(self.ima_bina)
    
    def firma_fig(self):
        self.im_2=self.ima_bina
        self.ima_b=np.zeros(self.ima_bina.shape)
        for fil in range(1,self.ima_bina.shape[0]-1,1):
            for col in range(1,self.ima_bina.shape[1]-1,1):
                if (self.im_2[fil,col]==1)and(self.im_2[fil,col+1]==0 or self.im_2[fil+1,col]==0 or 
                                       self.im_2[fil,col-1]==0 or self.im_2[fil-1,col]==0):
                    self.ima_b[fil,col]=1
        
        
        imasec = self.ima_b
        plt.figure()
        plt.imshow(imasec)
        coordenadas = []
        total_pixeles = []
        datos_cor = [] #Matriz donde se guardan las coordenadas de la figura
        for fil in range(imasec.shape[0]):
            for col in range(imasec.shape[1]):
                if imasec[fil,col]==1:
                    datos_cor = []
                    i=fil
                    j=col
                    num_pixeles = 0 #total de pixeles que contiene la imagen por borde
                    imasec[i,j]=0 #matriz de ayuda para establecer el borde
                    stop = 0
                    while stop == 0:
                        if imasec[i,j+1]==1:
                            datos_cor.append(np.array([i,j])) #se guardan la fila y columna
                            num_pixeles += 1 #se aumenta el # de pixeles
                            imasec[i,j]=0 #se borra el pixel actual 
                            j=j+1
                        elif imasec[i+1,j+1]==1:
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                            i += 1
                            j += 1
                        elif imasec[i+1,j]==1:
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                            i += 1
                        elif imasec[i+1,j-1]==1:
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                            i += 1
                            j -= 1
                        elif imasec[i,j-1]==1:
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                            j -= 1
                        elif imasec[i-1,j-1]==1:
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                            i -= 1
                            j -= 1
                        elif imasec[i-1,j]==1:
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                            i -= 1
                        elif imasec[i-1,j+1]==1:
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                            i -= 1
                            j += 1
                        else:
                            stop = 1
                            datos_cor.append(np.array([i,j]))
                            num_pixeles += 1
                            imasec[i,j]=0
                    # plt.figure()
                    # plt.imshow(imasec)
                    coordenadas.append(datos_cor) #se guarda la lista de arrays por figura
                    total_pixeles.append(num_pixeles) # se guarda el # total de pixeles por figura
        
        print(len(coordenadas))
        
        self.firmas_totales = [] #matriz para guardar la firma de cada figura encontrada
        self.centroides=[]
        for i in range(len(coordenadas)):
            total_filas = 0
            total_columnas = 0
            punto_x = 0
            punto_y = 0
            for j in range(len(coordenadas[i])):
                total_filas += coordenadas[i][j][0]
                total_columnas += coordenadas[i][j][1]
        
            punto_x=total_columnas/total_pixeles[i] #punto x para el cenotride
            punto_y=total_filas/total_pixeles[i]  #punto y para el centroide
            # print(punto_x)
            # print(punto_y)
            self.centroides.append(np.array([punto_x,punto_y]))
            
            self.firma=[]
            for dist in range(len(coordenadas[i])):
                distancia = np.sqrt((punto_x - coordenadas[i][dist][1])**2 +
                                (punto_y - coordenadas[i][dist][0])**2)
                self.firma.append(distancia)
            self.firma = self.firma/np.max(self.firma) #normalizamos la gráfica de firma
            # plt.figure()
            # plt.plot(self.firma)
            firma_suavizada = scp.savgol_filter(self.firma,97,6) #se suaviza la gráfica
            plt.figure()
            plt.plot(firma_suavizada)
            self.firmas_totales.append(firma_suavizada)
            
        #Obtener las características de la gráfica
        picos_totales = []
        for firm in range(len(self.firmas_totales)):
            picos = 0
            alcance = len(self.firmas_totales[firm])
            for dat in range(alcance):
                now = self.firmas_totales[firm][dat]
                if dat == 0:
                    picos += 1
                elif (dat < alcance-1 and self.firmas_totales[firm][dat-1] < now and 
                      self.firmas_totales[firm][dat+1] < now):
                    picos += 1
                elif dat == alcance-1:
                    picos += 1
            picos_totales.append(picos)
            print(picos)
            
        cuadrados = 0
        triangulos = 0
        circulos = 0
    
        indices_cuadrados=[]
        indices_triangulos=[]
        indices_circulos=[]
        indices_totales=[]
        for i in range(len(picos_totales)):
            if picos_totales[i] >= 5 and picos_totales[i] < 8:
                cuadrados += 1
                indices_cuadrados.append(i)
            elif picos_totales[i] >= 2 and picos_totales[i] < 5:
                triangulos += 1
                indices_triangulos.append(i)
            else:
                circulos += 1
                indices_circulos.append(i)
        indices_totales.append(indices_cuadrados)
        indices_totales.append(indices_triangulos)
        indices_totales.append(indices_circulos)
        print('El total de cuadrados es: '+str(cuadrados))
        print('El total de triangulos es: '+str(triangulos))
        print('El total de circulos es: '+str(circulos))
        
        self.centro_x=0
        self.centro_y=0
        if self.forma=='cuadrado':
            for dd in range(len(indices_cuadrados)):
                indice_forma=indices_cuadrados[dd]
                self.centro_x=self.centroides[indice_forma][0]
                self.centro_y=self.centroides[indice_forma][1]
                print(self.centro_x)
                print(self.centro_y)
        elif self.forma=='triangulo':
            for dd in range(len(indices_triangulos)):
                indice_forma=indices_triangulos[dd]
                self.centro_x=self.centroides[indice_forma][0]
                self.centro_y=self.centroides[indice_forma][1]
                print(self.centro_x)
                print(self.centro_y)
        elif self.forma=='circulo':
            for dd in range(len(indices_circulos)):
                indice_forma=indices_circulos[dd]
                self.centro_x=self.centroides[indice_forma][0]
                self.centro_y=self.centroides[indice_forma][1]
                print(self.centro_x)
                print(self.centro_y)
                
        self.movimiento_servos()
        
    def movimiento_servos(self):
        pos1=self.centro_y
        pos2=self.centro_x
        
        ####Relacion entre distancia y ángulo
        a_iniy=62
        a_fiy=102
        a_inix=60
        a_finx=110
        
        res1=round((pos1*50)/1266)
        res2=round((pos2*40)/956)
        
        posicion1=a_iniy+(res2)
        posicion2=a_finx-(res1-3)
        
        if posicion1 >a_fiy:
            restita=abs(posicion1-a_fiy)
            posicion1=posicion1-restita
        elif posicion2 >a_finx:
            restita2=abs(posicion2-a_finx)
            posicion2=posicion2-restita2
        else:
            posicion1=a_iniy+res2
            posicion2=a_inix+res1
        
        
        board.digital[pinS1].write(posicion1)
        time.sleep(2)
        board.digital[pinS2].write(posicion2)
        time.sleep(2)
        
    def nueva_w(self):
        self.vent1 = tk.Toplevel(self.master)
        self.vent1.resizable(0, 0)  
        self.vent1.geometry('960x700')
        self.imagen3 = tk.PhotoImage(file="boton2.png")
        self.boton2 = tk.Button(
            self.vent1,
            image=self.imagen3,
            font=("Helvetica", 10, "bold"),
            justify="center",
            height=70,
            width=200,
            borderwidth=5,
            command=self.apagar,
        )
        self.boton2.place(x=200, y=600)
        self.panel2 = tk.Label(self.vent1)
        self.panel2.config(width=960, height=540)
        self.panel2.place(x=0,y=0)       
        self.url2='http://192.168.0.13:4747/video.jpg'        
        self.camara2=cv.VideoCapture(self.url2)        
        self.camara2.set(cv.CAP_PROP_FRAME_WIDTH, camara_x)
        self.camara2.set(cv.CAP_PROP_FRAME_HEIGHT, camara_y)
        self.visualizar()
       
    def visualizar(self):
        _, frame2 = self.camara2.read()
        frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
        frame2 = Image.fromarray(frame2)
        frame2 = ImageTk.PhotoImage(frame2)        
        self.panel2.config(image=frame2)
        self.panel2.image=frame2
        self.panel.after(1, self.visualizar)
    
    def apagar(self):
        self.camara2.release()
        self.vent1.destroy()
        
        
raiz = tk.Tk()
prin = MenuPrincipal(raiz)
prin.mainloop()
board.exit()


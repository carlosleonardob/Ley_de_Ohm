# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 20:20:24 2024

@author: UIS
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit as cf
import matplotlib.pyplot as plt
import streamlit as st
@st.cache_data
@st.cache_resource

def Fiteo(x,y,p_ini):
    bounds=((0,0),(np.inf,1))
    best_val,cov=cf(N_Ohm,x,y,p0=p_ini,method='trf',
                    bounds=bounds)
    return best_val

def N_Ohm(v,G,r):
    return G*((v)**r)

st.write('Ley de Ohm')
uploaded_file =  st.file_uploader("Choose a file", 
                                  type=['csv', 'xlsx', 'txt'])

if uploaded_file is not None:
    # Check the file type
    file_type = uploaded_file.type

    # Use the appropriate pandas function to read the file
    if file_type == "text/csv":
        df = pd.read_csv(uploaded_file,delimiter=';')
        
    elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        df = pd.read_excel(uploaded_file)
        data = pd.read_excel(uploaded_file, sheet_name=None)
        
        sheet_name = st.selectbox("Choose a sheet", list(data.keys()))
        
        df = data[sheet_name]
        
    elif file_type == "text/plain":
        df = pd.read_csv(uploaded_file, sep="\t")  # assuming tab-separated values

    # Use the dataframe in your app
  

df = df.dropna()  # Remove rows with missing values
    
datos=df.values #matrix de datos a trabajar
st.write(df) #muestra los datos de la hoja de excel


datos=df.values
V=datos[:,0] #voltaje
I=datos[:,1] #corriente

#descripción del modelo: ecuacion tipo latex
st.text('El modelo físico se basa en la siguiente ecuación')
st.latex(r'I=G\,V^r')


st.text('Ingrese lo valores considerados para los siguientes parametros')
G = st.number_input('Enter parameter G:')
r= st.number_input('Enter parameter r:')

p_ini=[G,r] #G,r
best_val=Fiteo(V,I,p_ini)
print('best_val=',best_val)
V_val=np.linspace(V[0], V[-1],num=200)
G=best_val[0]
r=best_val[1]
I_val=N_Ohm(V_val, G, r)

fig, ax = plt.subplots(figsize=(3, 3))
ax.plot(V,I,'*',c='r',label='datos')
ax.plot(V_val,I_val,ls=':',label='modelo')
ax.set_xlabel('voltaje (V)')
ax.set_ylabel('Corriente (mA)')
ax.set_title('Caracteristica I vs V para un bombillo')
st.pyplot(fig)

st.text('Valores optimizados para el modelo de acuerdo a los datos experimentales')
st.write('G=',"{:.3f}".format(G))
st.write('r=',"{:.3f}".format(r))

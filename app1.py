import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from PIL import Image

 


st.title("Profil professionnel")
# st.header("Profil professionnel")

col5,col6 = st.columns(2)
col5.metric("Nom","Affognon")
col6.metric("Prénom","Augustin")


col7,col8 = st.columns(2)



col7.metric("Profession","Dev IA")
col8.metric("Année","2024")

sidebarlogo = Image.open('MainImg1.jpg').resize((300, 300))
st.image(sidebarlogo, use_container_width='auto')

profession = "<h4> Compétences : </h4>  <ul> <li> Java </li> <li> Python </li> <li> React </li> </ul> "

st.markdown(profession,True)
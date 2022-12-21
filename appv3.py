import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
st.set_page_config(page_title = "Real Estate Sale Price Prediciton",
                   layout = "wide")

header = st.container()

Capstone_Project, Exploratives, House_Price = st.tabs([ "Our Capstone Project", "Descriptive Analysis", "House Price Prediction"])

with header:
    st.title("Welcome to our Project!")
    st.text("In this project we create a tool to dynamically predict Saleprices of Real Estate")
    
with Exploratives:
    #Überschrift
    st.header("Kaggle Competition: House Prices")
    #load data-this should be cached later
    @st.cache
    def read(file):
        data = pd.read_csv(file)
        return data
    df = read("train.csv")
    st.subheader("Distribution of Target Variable in training set")
    Saleprices = pd.DataFrame(df["SalePrice"].value_counts())
    fig_SalePrice = px.histogram(df, x="SalePrice")
    st.plotly_chart(fig_SalePrice, use_container_width=True)
with House_Price:
    with st.form(key='my_form'):
        st.header("Please select values")
        c1, c2, c3, c4, c5 = st.columns(5)
        u_qual = c1.slider("Quality of home", 1,10,5)
        u_liv = c2.number_input("Gross living area (sq. feet)", min_value = int(df["GrLivArea"].min()), 
                                max_value = int(df["GrLivArea"].max()), value = int(df["GrLivArea"].median()),
                                step = 20)
        u_car = c3.slider("Garage size (cars)", min_value = int(df["GarageCars"].min()), 
                                max_value = int(df["GarageCars"].max()), value = 2,
                                step = 1)
        u_bas1 = c4.number_input("Basement size total", min_value = int(df["TotalBsmtSF"].min()), 
                                max_value = int(df["TotalBsmtSF"].max()), value = int(df["TotalBsmtSF"].median()), 
                                step = 20)
        u_bas2 = c5.number_input("Basement size finished", min_value = int(df["BsmtFinSF1"].min()), 
                                max_value = int(df["BsmtFinSF1"].max()), value = int(df["BsmtFinSF1"].median()),
                                step = 20)
            
        user_input = pd.DataFrame({
            "OverallQual": [u_qual],
            "GarageCars": [u_car],
            "GrLivArea": [u_liv],
            "TotalBsmtSF": [u_bas1],
            "BsmtFinSF1": [u_bas2]
        })
        submit_button = st.form_submit_button(label='Submit')
    #st.write(user_input)
        
            
    
    model_red = pickle.load(open('model_redv2.pkl', 'rb'))
    predicted_Value = model_red.predict(user_input).item()
    st.markdown("**Your Valuation:**")
    predicted_Value = round(predicted_Value)
    MAPE = 0.13462876068606142
    range_max = round(predicted_Value * (1+MAPE))
    range_min = round(predicted_Value *(1-MAPE))
    st.subheader(f"${predicted_Value}")
    st.write(f"${range_min} - {range_max}")
    if predicted_Value > 400000:
        st.warning("your home appears to be valued higher than 98% of our Training set for such high-value homes this prediction might not be accurate", icon = "🚨")
    #Quick fix prescription, Ausbau Basement
    user_input_ext = user_input.copy()
    user_input_ext_combined = user_input.copy()
    predictions = pd.DataFrame({
        "improvement": [],
        "potential": []
    })
    if user_input["BsmtFinSF1"].item() < (0.8 * user_input["TotalBsmtSF"].item()):
        user_input_ext_combined["BsmtFinSF1"] = user_input_ext_combined["TotalBsmtSF"] *0.8
        Improvementpot = round(model_red.predict(user_input_ext_combined).item() - predicted_Value)
        #st.markdown("**by finishing your basement you could increase SalesPrice by:**")
        #st.subheader(f"${Improvementpot}")
        predictions = pd.DataFrame({
            "improvement": ["finishing Basement"],
            "potential": [Improvementpot]
        })
        
    #Not so quick: Improving Overall Qual and expanding Garage
    step_increment = ["OverallQual", "GarageCars"]
    prev = model_red.predict(user_input).item()
    for s in step_increment:
        user_input_ext[s] = user_input_ext[s] + 1
        user_input_ext_combined[s] = user_input_ext_combined[s] + 1
        pred = model_red.predict(user_input_ext).item()
        append = pd.DataFrame(
        {
            "improvement": [s],
            "potential": [pred - predicted_Value]
        })
        user_input_ext = user_input.copy()
        predictions = pd.concat([predictions, append])
    combined = pd.DataFrame(
        {
        "improvement": ["combined"],
        "potential": [model_red.predict(user_input_ext_combined).item() - predicted_Value]
        })
    predictions = pd.concat([predictions, combined])
    #st.write(predictions)
            
    
with Exploratives:
    #plotting OverallQual with highlighting
    c = {}
    z = "blue"
    try: 
        yi = user_input["OverallQual"].item()
        keys = df["OverallQual"].unique().tolist()
        keys.remove(yi)
        c[yi] = "green"
    except:
        keys = df["OverallQual"].unique().tolist()
    for i in keys:
        c[i] = z
    fig_OverallQual = px.histogram(df, x="OverallQual", color=df["OverallQual"].tolist(), 
    color_discrete_map=c)
    fig_OverallQual.update_layout(showlegend=False)
    #plotting GrLivArea with Highlighting
    #first step creating bins
    data = df["GrLivArea"]
    data = pd.DataFrame(data)
    i = 399
    list_of_bins = []
    while i < 5700:
        list_of_bins.append(i)
        i += 100
    #transforming data to fit in bins
    data = np.where(data < list_of_bins[0], list_of_bins[0], data)
    for i in range(len(list_of_bins)):
        data = np.where((data < list_of_bins[i]) & (data > list_of_bins[i-1]), list_of_bins[i], data)
    #transforming userinput to fit in bins
    try: 
        for i in range(len(list_of_bins)):
            if (user_input["GrLivArea"].item() < list_of_bins[i]) & (user_input["GrLivArea"].item() > list_of_bins[i-1]):
                user_input["GrLivArea"] = list_of_bins[i]
    except:
        ()
    #creating dict for color map
    c = {}
    z = "blue" # default color
    #try highlighting user input
    try: 
        yi = user_input["GrLivArea"].item()
        keys = np.unique(data).tolist()
        keys.remove(yi)
        c[yi] = "green"
    except:
        keys = np.unique(data).tolist() # if not defined
    for i in keys:
        c[i] = z
    #plotting
    data = pd.DataFrame(data)
    fig_GrLivArea = px.histogram(data, x=0, color=data[0], 
    color_discrete_map=c) 
    fig_GrLivArea.update_layout(showlegend=False)
    #Plotting Basement Total
    data = df["TotalBsmtSF"].loc[df["TotalBsmtSF"]< 4000]
    data = pd.DataFrame(data)
    i = 0
    list_of_bins = []
    while i < 3251:
        list_of_bins.append(i)
        i += 50
    #data = np.where(data < list_of_bins[0], list_of_bins[0], data)
    for i in range(len(list_of_bins)):
        data = np.where((data < list_of_bins[i]) & (data > list_of_bins[i-1]), list_of_bins[i], data)
    for i in range(len(list_of_bins)):
        if (user_input["TotalBsmtSF"].item() < list_of_bins[i]) & (user_input["TotalBsmtSF"].item() > list_of_bins[i-1]):
            user_input["TotalBsmtSF"] = list_of_bins[i]
    c = {}
    z = "blue"
    try: 
        yi = user_input["TotalBsmtSF"].item()
        keys = np.unique(data).tolist()
        keys.remove(yi)
        c[yi] = "green"
    except:
        keys = np.unique(data).tolist()
    for i in keys:
        c[i] = z
    data = pd.DataFrame(data)
    fig_TotalBsmtSF = px.histogram(data, x=0, color=data[0], 
    color_discrete_map=c)
    fig_TotalBsmtSF.update_layout(showlegend=False)
    #Plotting Basement finished
    data = df["BsmtFinSF1"].loc[df["BsmtFinSF1"]< 4000]
    data = pd.DataFrame(data)
    i = 0
    list_of_bins = []
    while i < 3251:
        list_of_bins.append(i)
        i += 50
    #data = np.where(data < list_of_bins[0], list_of_bins[0], data)
    for i in range(len(list_of_bins)):
        data = np.where((data < list_of_bins[i]) & (data > list_of_bins[i-1]), list_of_bins[i], data)
    for i in range(len(list_of_bins)):
        if (user_input["BsmtFinSF1"].item() < list_of_bins[i]) & (user_input["BsmtFinSF1"].item() > list_of_bins[i-1]):
            user_input["BsmtFinSF1"] = list_of_bins[i]
    c = {}
    z = "blue"
    try: 
        yi = user_input["BsmtFinSF1"].item()
        keys = np.unique(data).tolist()
        keys.remove(yi)
        c[yi] = "green"
    except:
        keys = np.unique(data).tolist()
    for i in keys:
        c[i] = z
    data = pd.DataFrame(data)
    fig_BsmtFinSF1 = px.histogram(data, x=0, color=data[0], 
    color_discrete_map=c)
    fig_BsmtFinSF1.update_layout(showlegend=False)
    #Garage Cars
    c = {}
    z = "blue"
    try: 
        yi = user_input["GarageCars"].item()
        keys = df["GarageCars"].unique().tolist()
        keys.remove(yi)
        c[yi] = "green"
    except:
        keys = df["GarageCars"].unique().tolist()
    for i in keys:
        c[i] = z
    fig_GarageCars = px.histogram(df, x="GarageCars", color=df["GarageCars"].tolist(), 
    color_discrete_map=c)
    fig_GarageCars.update_layout(showlegend=False)
    #plotting
    plot1, plot2, plot3 = st.columns(3)
    plot1.subheader("Distribution of Overall Quality")
    plot1.plotly_chart(fig_OverallQual, use_container_width=True)
    plot2.subheader("Gross living area Distribution")
    plot2.plotly_chart(fig_GrLivArea, use_container_width=True)
    plot3.subheader("Distribution of Garage Cars")
    plot3.plotly_chart(fig_GarageCars, use_container_width=True)
    #try pyplot for comparison
    plot4, plot5 = st.columns(2)
    plot4.subheader("Distribution of Basement sq. ft. (Total)")
    plot4.plotly_chart(fig_TotalBsmtSF, use_container_width=True)
    plot5.subheader("Distribution of Basement sq. ft. (finished)")
    plot5.plotly_chart(fig_BsmtFinSF1, use_container_width=True)
with House_Price:
  predictions[""] = 1
  predictions = pd.concat([predictions, append])
  fig = px.bar(predictions, x="", y="potential", color="improvement", title="Improvement Potential")
  st.plotly_chart(fig, use_container_width=True)
  
with Capstone_Project:
  st.header("Capstone Project")
  st.write("__RandomForest__")
  st.write("training the data with all the variables")
  
  data = [
    ["R2 on train", 0.9735],
    ["RMSE", 27092.5458],
    ["MAPE", 0.1164],
    ["R2 on test", 0.8972]]
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("we want to know about the most important variables for our prediction")
  image = Image.open('variable_importance_randomforest-2.png')
  st.image(image, caption='RandomForest_variable_importance')
  
  st.write("training the data with just the five most important variables")
  st.write("new performance")
  data = [
    ["R2 on train", 0.9705],
    ["RMSE", 27219.9228],
    ["MAPE", 0.1332],
    ["R2 on test", 0.8962]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("absolute changes")
  data = [
    ["RMSE", -32.3219],
    ["MAPE", 0.0194],
    ["R2 on test", 0.0002]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("percent changes")
  data = [
    ["RMSE", -0.0012],
    ["MAPE", 0.1705],
    ["R2 on test", 0.0003]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  
  
  
  
  
  
  st.write("__RandomForest with Hypertuning__")
  st.write("training the data with all the variables")
  
  data = [
    ["R2 on train", 0.9678],
    ["RMSE", 27093.0079],
    ["MAPE", 0.1151],
    ["R2 on test", 0.8972]]
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("we want to know about the most important variables for our prediction")
  #image = Image.open('variable_importance_randomforest_hype.png')
  #st.image(image, caption='RandomForest_with hyperparameter_variable_importance')
  
  st.write("training the data with just the five most important variables")
  st.write("new performance")
  data = [
    ["R2 on train", 0.9453],
    ["RMSE", 31899.4466],
    ["MAPE", 0.1460],
    ["R2 on test", 0.8264]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("absolute changes")
  data = [
    ["RMSE", 4647.2019],
    ["MAPE", 0.0309],
    ["R2 on test", -0.0696]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("percent changes")
  data = [
    ["RMSE", 0.1774],
    ["MAPE", 0.2688],
    ["R2 on test", -0.0777]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  
  
  
  
  
  
  
    
  st.write("__RandomForest mit Hypertuning und randomizedgridsearch__")
  st.write("training the data with all the variables")
  
  data = [
    ["R2 on train", 0.9611],
    ["RMSE", 17392.009],
    ["MAPE", 0.1148],
    ["R2 on test", 0.8891]]
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("we want to know about the most important variables for our prediction")
  #image = Image.open('variable_importance_randomforest_hype_rand.png')
  #st.image(image, caption='RandomForest_with hyperparameter and randomizedgridsearch_variable_importance')
  
  st.write("training the data with just the five most important variables")
  st.write("new performance")
  data = [
    ["R2 on train", 0.9561],
    ["RMSE", 27409.8859],
    ["MAPE", 0.1333],
    ["R2 on test", 0.8948]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("absolute changes")
  data = [
    ["RMSE", -726.9501],
    ["MAPE", 0.0185],
    ["R2 on test", -0.0898]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("percent changes")
  data = [
    ["RMSE", -0.0258],
    ["MAPE", 0.1609],
    ["R2 on test", 0.0064]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  
  
  
  
  
  
  st.write("__LinearRegression__")
  st.write("our results when just training the data with all the variables")
  
  data = [
    ["R2 on train", 0.9327],
    ["RMSE", 28856.0652],
    ["MAPE", 0.1183],
    ["R2 on test", 0.8834]]
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("we want to know about the most important variables for our prediction")
  #image = Image.open('variable_importance_randomforest_hype_rand.png')
  #st.image(image, caption='RandomForest_with hyperparameter and randomizedgridsearch_variable_importance')
  
  st.write("training the data with just the five most important variables")
  st.write("new performance")
  data = [
    ["R2 on train", 0.0481],
    ["RMSE", 85029.4914],
    ["MAPE", 0.4082],
    ["R2 on test", -0.0126]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("absolute changes")
  data = [
    ["RMSE", 57777.2467],
    ["MAPE", 0.2944],
    ["R2 on test", -0.8959]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("percent changes")
  data = [
    ["RMSE", 2.1201],
    ["MAPE", 2.5866],
    ["R2 on test", -1.0142]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  
  
  
  
  st.write("__DecisionTree__")
  st.write("our results when just training the data with all the variables")
  
  data = [
    ["R2 on train", 1.0],
    ["RMSE", 50940.8499],
    ["MAPE", 0.2194],
    ["R2 on test", 0.0023]]
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("we want to know about the most important variables for our prediction")
  #image = Image.open('variable_importance_decisiontree.png')
  #st.image(image, caption='decisiontree_variable_importance')
  
  st.write("training the data with just the five most important variables")
  st.write("new performance")
  data = [
    ["R2 on train", 1.0],
    ["RMSE", 62088.6109],
    ["MAPE", 0.2757],
    ["R2 on test", 0.0045]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("absolute changes")
  data = [
    ["RMSE", 11147.7610],
    ["MAPE", 0.0563],
    ["R2 on test", 0.0023]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("percent changes")
  data = [
    ["RMSE", 0.2188],
    ["MAPE", 0.2565],
    ["R2 on test", -0.9948]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  
  
  
  
  
  
  st.write("__LassoModel__")
  st.write("our results when just training the data with all the variables")
  
  data = [
    ["R2 on train", 0.9327],
    ["RMSE", 28135.8458],
    ["MAPE", 0.1157],
    ["R2 on test", 0.8891]]
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("we want to know about the most important variables for our prediction")
  #image = Image.open('variable_importance_randomforest_hype_rand.png')
  #st.image(image, caption='RandomForest_with hyperparameter and randomizedgridsearch_variable_importance')
  
  st.write("training the data with just the five most important variables")
  st.write("new performance")
  data = [
    ["R2 on train", 0.0481],
    ["RMSE", 85029.4914],
    ["MAPE", 0.4082],
    ["R2 on test", -0.0126]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("absolute changes")
  data = [
    ["RMSE", 57777.2467],
    ["MAPE", 0.2944],
    ["R2 on test", -0.8959]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("percent changes")
  data = [
    ["RMSE", 2.1201],
    ["MAPE", 2.5866],
    ["R2 on test", -1.0142]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  
  
  
  
    
  

       

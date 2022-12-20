import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
st.set_page_config(page_title = "Real Estate Sale Price Prediciton",
                   layout = "wide")

header = st.container()

Exploratives, House_Price = st.tabs(["Descriptive Analysis", "House Price Prediction"])

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
with House_Price:
    with st.form(key='my_form'):
        st.header("Please select values")
        c1, c2, c3, c4, c5 = st.columns(5)
        u_qual = c1.slider("Quality of home", 1,10,5)
        u_liv = c2.number_input("Gross living area (sq. feet)", min_value = round(df["GrLivArea"].min()), 
                                max_value = round(df["GrLivArea"].max()), value = round(df["GrLivArea"].median()),
                                step = 20)
        u_car = c3.slider("Garage size (cars)", min_value = df["GarageCars"].min(), 
                                max_value = df["GarageCars"].max(), value = 2,
                                step = 1)
        u_bas1 = c4.number_input("Basement size total", min_value = round(df["TotalBsmtSF"].min()), 
                                max_value = round(df["TotalBsmtSF"].max()), value = round(df["TotalBsmtSF"].median()), 
                                step = 20)
        u_bas2 = c5.number_input("Basement size finished", min_value = round(df["BsmtFinSF1"].min()), 
                                max_value = round(df["BsmtFinSF1"].max()), value = round(df["BsmtFinSF1"].median()),
                                step = 20)
            
        user_input = pd.DataFrame({
            "OverallQual": [u_qual],
            "GrLivArea": [u_liv],
            "GarageCars": [u_car],
            "TotalBsmtSF": [u_bas1],
            "BsmtFinSF1": [u_bas2]
        })
        submit_button = st.form_submit_button(label='Submit')
    st.write(user_input)
        
            
    
    model_red = pickle.load(open('model_red.pkl', 'rb'))
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
        st.markdown("**by finishing your basement you could increase SalesPrice by:**")
        st.subheader(f"${Improvementpot}")
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
    st.write(predictions)
            
    
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
    
    
with Exploratives:
    plot1, plot2, plot3 = st.columns(3)
    plot1.subheader("Distribution of Overall Quality")
    plot1.plotly_chart(fig_OverallQual)
    plot2.subheader("Gross living area Distribution")
    plot2.plotly_chart(fig_GrLivArea)
    plot3.subheader("Distribution of Garage Cars")
    plot3.plotly_chart(fig_GarageCars)
    #try pyplot for comparison
    plot4, plot5 = st.columns(2)
    plot4.subheader("Distribution of Basement sq. ft. (Total)")
    plot4.plotly_chart(fig_TotalBsmtSF)
    plot5.subheader("Distribution of Basement sq. ft. (finished)")
    plot5.plotly_chart(fig_BsmtFinSF1)

        
        
        
        
        
        
        
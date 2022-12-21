import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
st.set_page_config(page_title = "Real Estate Sale Price Prediciton",
                   layout = "wide")

header = st.container()

Capstone_Project, Exploratives, House_Price, Alex = st.tabs([ "Our Capstone Project", "Descriptive Analysis", "House Price Prediction", "Alex"])

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
    st.subheader("Distribution of Sale Prices in Ames, Iowa")
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
  st.write("🎯<b>Our goal:</b> predicting sale price of Real Estate in Ames, Iowa: <br>With this we aim to increase transparency in the complex real-estate market. <br>Real Estate Brokers are very expensive and with this tool customers can quickly and cheaply replace the cost-intense valuation", unsafe_allow_html = True)
  st.subheader("Selected Data")
  st.write(df.head(10))
  numeric_features = []
  cat_features = []
  for col in df.drop("SalePrice", axis = 1).columns:
    if df[col].dtypes != df["CentralAir"].dtypes:
      numeric_features.append(col)
    else:
        cat_features.append(col)
  sns.set_theme(style="ticks")
  # Generate a large random dataset
  corr = df[numeric_features].corr()
  rs = np.random.RandomState(33)
  mask = np.triu(np.ones_like(corr, dtype=bool))
  fig, axs = plt.subplots(figsize=(45, 25))
  sns.heatmap(corr , annot=True, square=True, mask=mask,  cmap="Oranges")
  plt.title('Training Set Correlations', size=15)
  st.pyplot(fig)
  st.subheader("Imputation")
  st.markdown("1. Drop all columns with >90% Na Values. => 2. KNN Imputation for numerical values => 3. Mode Imputation for categorical values.")

  
  st.subheader("Training our Model")
  st.write("We used the following machine learning algorithms")      
  
  data = [["RandomForest", "RandomForest with Hypertuning", "RandomForest with randomizedgridsearch and Hypertuning", "LinearRegression", "DecisionTree", "LassoModel"]]
  
  st.write("Why did we decide to use several machine learning algorithms?")
  st.write("As the choice of the perfect algoritm depends on the need of a project and the characteristics of the data, we had to try out several algorithms to find the one that works the best for our specific problem.")
  
  df = pd.DataFrame(data)

  st.dataframe(df)
  
  st.write("__RandomForest__")
  st.write("training the data with all variables")
    
  
  
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
  
  st.write("percentage changes")
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
  image = Image.open('variable_importance_randomforest_hype.png')
  st.image(image, caption='RandomForest_with hyperparameter_variable_importance')
  
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
  
  st.write("percentage changes")
  data = [
    ["RMSE", 0.1774],
    ["MAPE", 0.2688],
    ["R2 on test", -0.0777]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  
  
  
  
  
  
  
    
  st.write("__RandomForest with randomizedgridsearch and Hypertuning__")
  st.write("training the data with all the variables")
  
  data = [
    ["R2 on train", 0.9611],
    ["RMSE", 17392.009],
    ["MAPE", 0.1148],
    ["R2 on test", 0.8891]]
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("we want to know about the most important variables for our prediction")
  image = Image.open('variable_importance_randomforest_hype_rand.png')
  st.image(image, caption='RandomForest_with hyperparameter and randomizedgridsearch_variable_importance')
  
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
  
  st.write("percentage changes")
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
  image = Image.open('variable_importance_randomforest_hype_rand.png')
  st.image(image, caption='RandomForest_with hyperparameter and randomizedgridsearch_variable_importance')
  
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
  
  st.write("percentage changes")
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
  image = Image.open('variable_importance_decisiontree.png')
  st.image(image, caption='decisiontree_variable_importance')
  
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
  
  st.write("percentage changes")
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
  image = Image.open('variable_importance_randomforest_hype_rand.png')
  st.image(image, caption='Lasso_variable_importance')
  
  st.write("training the data with just the five most important variables")
 
  
  st.write("absolute changes")
  data = [
    ["RMSE", 57777.2467],
    ["MAPE", 0.2944],
    ["R2 on test", -0.8959]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.write("percentage changes")
  data = [
    ["RMSE", 2.1201],
    ["MAPE", 2.5866],
    ["R2 on test", -1.0142]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])

  st.dataframe(df)
  
  st.subheader("Presenting our best model - The RandomForest")
  st.markdown("After all machine learning algorithms we used, we got the best performance (while just using 5 variables) of predicting the saleprice of the house with the RandomForest")
  st.markdown("Below there is a scatterplot which presents the performance of our model:")
  
  image = Image.open('scatterplot-4.png')
  st.image(image, caption='Performance of RandomForest')
  
  
  st.write("The five most important feature variables are the following")
  
  data = [
    [0.51, "OverallQual", "Overall material and finish quality"],
    [0.11, "GrLivArea", "Above grade (ground) living area square feet"],
    [0.05, "GarageCars", "Size of garage in car capacity"],
    [0.04, "BsmtFinSF1", "Type 1 finished square feet"],
    [0.03, "TotalBsmtSF", "Total square feet of basement area"]
  ]
  df = pd.DataFrame(data, columns=["feature_importance", "variable", "meaning_of_the_variable"])

  st.dataframe(df)
  
  st.markdown("__Conclusion and usage of results__")
  
  st.write("performance numbers")

  data = [
    ["R2 on train", 0.9705],
    ["RMSE", 27219.9228],
    ["MAPE", 0.1332],
    ["R2 on test", 0.8962],
    ["MAE", 22473.7433]]
  
  df = pd.DataFrame(data, columns=["type of measure", "value"])
  
  st.dataframe(df)
  
  st.write("Conclusion of our performance numbers")
  
  st.write("further usage")

with House_Price:
  st.subheader('Forecast of Salesprice per sqft')
  image = Image.open('houe_sqft_price_usa.png')
  st.image(image, caption='Forecast of Salesprice per sqft')
  
with Alex:
  df = read('City_time_series.csv')
  nan_percent = df.isnull().mean()
  # select the columns with less than 75% NaN values
  df = df[nan_percent[nan_percent < 0.75].index]
  mid = pd.DataFrame()
mid['date']=df.Date
df.drop('Date', axis=1, inplace=True)
mid.reset_index()
mid['date'] = pd.to_datetime(mid['date'], format='%Y-%m-%d')
mid['date_obj'] = mid['date'].dt.date
df.drop('RegionName', axis=1, inplace=True)
df['ZHVIPerSqft_AllHomes'].interpolate(method='polynomial', order=3, inplace=True)
df['ZHVI_BottomTier'].interpolate(method='polynomial', order=3, inplace=True)
df['ZHVI_MiddleTier'].interpolate(method='polynomial', order=3, inplace=True)
df['ZHVI_TopTier'].interpolate(method='polynomial', order=3, inplace=True)   
df_zhvi=pd.DataFrame()
df_zhvi['price_sqft']=df.ZHVIPerSqft_AllHomes
df_zhvi['bottom']=df.ZHVI_BottomTier
df_zhvi['middle']=df.ZHVI_MiddleTier
df_zhvi['top']=df.ZHVI_TopTier
df=df_zhvi
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
mid=mid.iloc[2:]
mid=mid.reset_index()
df['date']=mid.date_obj
df=df.groupby('date').mean()
df['date']=df.index
df["date"] = pd.to_datetime(df["date"])
df = df.assign(year=df["date"].dt.year, month=df["date"].dt.month)

# Add widgets to allow the user to select a year and month and input the number of sqft
year = st.sidebar.slider("Select a year", min_value=1996, max_value=2017, value = 2008, step=1)
month = st.sidebar.slider("Select a month", min_value=1, max_value=12, value=8, step=1)
sqft = st.sidebar.number_input("Enter the number of sqft", value=1000)

# Select the data point for the selected year and month
selected_date = pd.to_datetime(f"{year}-{month}")
selected_point = df[(df.year == year) & (df.month == month)]

df['date']=df.index
df["date_dt"] = pd.to_datetime(df["date"])
df= df.assign(year=df["date_dt"].dt.year, month=df["date_dt"].dt.month).drop("date", axis=1)

df['ordinal'] = df.index.map(pd.Timestamp.toordinal)
model = LinearRegression()

# extract x and y from dataframe data
x = df[['ordinal']]
y = df[['price_sqft']]
from sklearn.preprocessing import PolynomialFeatures

# create a PolynomialFeatures object with degree 4
poly_features = PolynomialFeatures(degree=4)

# transform the ordinal feature into a polynomial feature
x_poly = poly_features.fit_transform(x)

# fit the model using the polynomial features
model = LinearRegression()
model.fit(x_poly, y)

# predict y values for the original x values using the polynomial model
y_pred = model.predict(x_poly)
  
ax1 = df.plot(y='price_sqft', c='r', figsize=(15, 6), grid=True, legend=False)
ax1.plot(df.date_dt, y_pred, label='Polynomial Model of Degree 4', c='blue')
# Add the selected data point to the plot
ax1.scatter(selected_date, selected_point.price_sqft, c="green", label="Selected Point", s=100)
plt.title("House Price per Square Foot USA")
plt.grid('auto')
plt.xlabel("Date")
plt.ylabel("Price")

predicted_price = sqft * float(selected_point.price_sqft)
print(predicted_price)

import calendar
# Determine the last day of the month
last_day = calendar.monthrange(year, month)[1]

# Create a Timestamp object using the year, month, and last day of the month
selected_date = pd.to_datetime(f'{year}-{month}-{last_day}')
df.reset_index(inplace=True, drop=True)

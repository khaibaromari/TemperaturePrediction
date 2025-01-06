import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

# load the dataset
df = pd.read_csv('./input/yearly_temp_merged_df.csv', index_col=0)

# rename column names
column_names = {
    "Area Code (FAO)":"area_code",
    "Year" : "Year",
    "Annual CO₂ emissions growth (%)" : "co2_emissions_rate",
    "Population growth (annual %)" : "pop_growth_rate",
    "Annual methane emissions in CO₂ equivalents" : "methane_emissions",
    "Annual nitrous oxide emissions in CO₂ equivalents" : "nitrous_oxide_emissions",
    "GDP (in USD)" : "gdp_usd"
    }

df.rename(columns=column_names, inplace=True)

# sidebar
st.sidebar.title('Table of contents')

# adding page navigation
pages=["👋 Introduction", "🔍 Data Exploration", "📊 Data Visualization", "🧩 Modeling", "🔮 Prediction", "📌 Conclusion"]
page=st.sidebar.radio("Go to", pages)

if page=="👋 Introduction":
    st.title('👋 Introduction')
    # Content of the Introduction
    st.write("""
    Climate change has become one of the most pressing challenges of the 21st century, 
    profoundly impacting ecosystems, economies, and communities worldwide. As global 
    temperatures continue to rise, understanding the interconnected factors contributing to this 
    phenomenon is critical for policymakers, researchers, and individuals alike. The "World 
    Temperature" project aims to delve into a comprehensive analysis of global temperature trends 
    and their relationships with key socioeconomic and environmental indicators. This initiative is 
    centered around the collection, exploration, and analysis of data to unravel patterns and derive 
    meaningful insights about our changing world.
    """)

    # Section: Objectives
    st.subheader("Objectives")
    st.write("""
    The primary goal of this project is to analyze global temperature trends in relation to key 
    variables that impact or are impacted by climate change. By leveraging data collected across 
    years and countries, we aim to explore the relationships between global temperatures and 
    indicators such as:
    - Annual CO₂ emissions growth (percentage)
    - Population growth (annual percentage)
    - Annual methane emissions in CO₂ equivalents
    - Annual nitrous oxide emissions in CO₂ equivalents
    - GDP in U.S. dollars

    Through this analysis, we seek to understand how economic growth, population dynamics, and 
    greenhouse gas emissions influence temperature changes. Additionally, the project aims to 
    identify disparities between developed and developing nations, providing insights into global 
    inequalities in climate impact and responsibility. 

    The findings are intended to serve as a foundation for sustainable development strategies and 
    evidence-based policymaking.
    """)


elif page=="🔍 Data Exploration":
    st.title('🔍 Data Exploration')
    st.write("""The data exploration page provides an initial overview of the dataset to understand its structure and key characteristics. Here, we display the first few rows of the dataset (head()), its dimensions (shape), and the data types of each column (dtypes). Additionally, we include summary statistics (describe()), highlighting essential metrics such as mean, standard deviation, and range for numerical columns. This exploration helps uncover patterns, identify potential anomalies, and prepare for deeper analysis.
    """)

    st.title("Dataset Overview")

    st.subheader("Preview of the First 10 Rows")
    st.write(df.head(10))

    st.subheader("Shape of the Dataset")
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

    st.subheader("Data Types of Each Column")
    st.write(df.dtypes)

    st.subheader("Descriptive Statistics")
    st.write(df.describe())
    
    st.subheader("Dataset Column Descriptions")

    st.write("""
    The dataset contains the following columns:

    - **Area Code (FAO)**: Integer codes representing specific regions or countries.  
    - **Area**: Names of countries or regions.  
    - **Year**: The year of the recorded data.  
    - **Value**: Temperature-related values, possibly yearly averages or deviations.  
    - **Annual CO₂ emissions growth (%)**: Percentage growth in annual CO₂ emissions.  
    - **Population growth (annual %)**: Annual population growth percentage.  
    - **Annual methane emissions in CO₂ equivalents**: Methane emissions measured in CO₂ equivalent units.  
    - **Annual nitrous oxide emissions in CO₂ equivalents**: Nitrous oxide emissions measured in CO₂ equivalent units.  
    - **GDP (in USD)**: GDP values in USD.  
    - **Population**: Total population of the area.  
    - **Continent**: Continent to which the area belongs.  
    """)
    
elif page=="📊 Data Visualization":
    st.title('📊 Data Visualization')
    st.dataframe(df.head(7))
    
    europe = [
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina",
    "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia",
    "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands", "North Macedonia",
    "Norway", "Poland", "Portugal", "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City"
        ]
    asian_countries = [
    "Afghanistan", "Armenia", "Azerbaijan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Cambodia",
    "China", "Cyprus", "Georgia", "India", "Indonesia", "Iran", "Iraq", "Japan", "Jordan",
    "Kazakhstan", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia",
    "Myanmar", "Nepal", "North Korea", "Oman", "Pakistan", "Palestine", "Philippines", "Qatar",
    "Saudi Arabia", "Singapore", "South Korea", "Sri Lanka", "Syria", "Taiwan", "Tajikistan",
    "Thailand", "Timor-Leste", "Turkey", "Turkmenistan", "United Arab Emirates", "Uzbekistan",
    "Vietnam", "Yemen"
    ]

    north_american_countries = [
    "Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba",
    "Dominica", "Dominican Republic", "El Salvador", "Grenada", "Guatemala", "Haiti",
    "Honduras", "Jamaica", "Mexico", "Nicaragua", "Panama", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States"
    ]

    south_american_countries = [
    "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana",
    "Paraguay", "Peru", "Suriname", "Uruguay", "Venezuela"
    ]

    australia_oceania_countries = [
    "Australia", "Fiji", "Kiribati", "Marshall Islands", "Micronesia", "Nauru",
    "New Zealand", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands",
    "Tonga", "Tuvalu", "Vanuatu"
    ]

    africa = [
    "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", "Cameroon",
    "Central African Republic", "Chad", "Comoros", "Democratic Republic of the Congo", "Djibouti",
    "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana",
    "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
    "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger",
    "Nigeria", "Republic of the Congo", "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles",
    "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia",
    "Uganda", "Zambia", "Zimbabwe"
    ]

    africa_df =  df[df['Area'].isin(africa)] ### Africa countries

    asia_df = df[df['Area'].isin(asian_countries)] ### Asian countries

    europe_df =  df[df['Area'].isin(europe)] ### Europe countries

    north_american_df =  df[df['Area'].isin(north_american_countries)] ### North American countries

    south_american_df =  df[df['Area'].isin(south_american_countries)] ### South American countries

    australia_df =  df[df['Area'].isin(australia_oceania_countries)] ### Australia and Oceania countries

    st.subheader('Line plot for the Temperature by Continents')
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Year', y='Value', data=asia_df,  color='blue',ci=None)
    sns.lineplot(x='Year', y='Value', data=africa_df,  color='red',ci=None)
    sns.lineplot(x='Year', y='Value', data=europe_df,  color='green',ci=None)
    sns.lineplot(x='Year', y='Value', data=north_american_df,  color='orange' , ci=None)
    sns.lineplot(x='Year', y='Value', data=south_american_df,  color='purple' , ci=None)
    #sns.lineplot(x='Year', y='Value', data=australia_df, color='gray',ci=None )
    plt.title('Temperature Change Over Time')
    plt.xlabel('Year')
    plt.ylabel('Temperature Change (°C)')
    plt.legend(['Asia', 'Africa', 'Europe', 'North America', 'South America', 'Australia and Oceania'])
    plt.grid(True)
    st.pyplot(plt)

    st.subheader('Heatmap of Correlation')
    # create  a data frame without object columns
    corr_df =  df.select_dtypes(exclude = 'object')
    ## create a corr matrics
    corr_matrix = corr_df.corr()
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap = 'seismic' , linewidths=0.5)
    st.pyplot(plt)


    st.header('Boxplot for Temperature anomaly and stats indicater')
    plt.figure(figsize=(14, 8))
    sns.boxplot(x="Year", y="Value", data=df, palette="coolwarm")
    plt.title("Temperature Change Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Temperature Change (°C)")
    plt.xticks(rotation=90)
    st.pyplot(plt)


    st.subheader('CO2 Categories')
    total_emissions = df['co2_emissions_rate'].dropna()
    total_emissions = df['co2_emissions_rate'] > 2
    total_emissions = df.groupby('Area')['co2_emissions_rate'].sum()
    total_emissions = total_emissions.sort_values(ascending=False).head(10)
    total_emissions = total_emissions[total_emissions > 1e-5]  # Adjust threshold as needed
    # Define a custom autopct function to skip very small percentages
    def custom_autopct(pct):
            return ('%1.1f%%' % pct) if pct > 0.5 else ''  # Display only if percentage > 0.5%
    # Creating the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(total_emissions, labels= total_emissions.index, autopct=custom_autopct, startangle=140)
    plt.title("Total CO₂ Emissions by Country (Aggregated Over 25 Years)")
    st.pyplot(plt)

elif page=="🧩 Modeling":
    st.title('🧩 Modeling')
    st.header('Objective')
    st.markdown('The objective of this project is to train different models based on number of features to predict the **temperature anaomaly** with respective to the given features.')
    st.subheader('Trained Models:')
    st.markdown('''
    1. Ridge Regression
    2. Random Forest Regressor
    3. Linear Regression
    4. Decision Tree Regressor
    5. Lasso Regressor''')

    st.divider()
    
    st.subheader('Steps taken to train the models:')
    st.markdown('''
    - Model instantiation
    - Model training based on 80%/20% train-test splits and stratified sampling based on the year column
    - Predictions on both test and train sets
    - Evaluation of model performance using appropriate metrics
    - Analysis of feature importance to understand the impact of each feature on the target variable
    - Visualization and analysis of results''')
    
    st.divider()
    
    st.subheader('Model Performance Metrics:')
    models_metrics = pd.read_excel('./input/models_metrics.xlsx', index_col=0)

    st.write("The following metrics were used to evaluate the models' performance:")
    st.markdown('''
    - R-squared (R2) score
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)''')

    st.write('Models Performance on training set:')
    # train test
    st.write(models_metrics.loc[:,models_metrics.columns.str.endswith('Train')], use_container_width=True)
    # test set
    st.write('Models Performance on test set:')
    st.write(models_metrics.loc[:,models_metrics.columns.str.endswith('Test')], use_container_width=True)

    st.divider()

    display_model_eval = st.button('Display Graphical Evaluation of Models Performance')
    if display_model_eval:
        st.subheader('Evaluation of Predictions vs Actual Cases')
        st.image('./input/prediction_actual_cases.png')

        st.subheader('Feature Importance')
        st.image('./input/feature_importance.png')

elif page=="🔮 Prediction":
    st.title('🔮 Prediction')
    st.header('Prediction Simulation with Ramdom Forest Regressor')

    # load the model
    import pickle
    with open('./model_pickled.pkl', 'rb') as f:
        model = pickle.load(f)

    # input features
    area = st.selectbox('Country', options=sorted(df['Area'].unique()), index=np.where(df.Area.unique() == 'Germany')[0].item())
    area_code = df[df['Area']==area]['area_code'].values[0]
    year = st.slider('Year', df['Year'].min(), df['Year'].max(), df['Year'].max())
    co2_rate = st.slider('Annual CO₂ emissions growth rate (%)', int(df['co2_emissions_rate'].min()), int(df['co2_emissions_rate'].max()), value=1599)  
    pop_growth_rate = st.slider('Population growth rate (%)', df['pop_growth_rate'].min(), df['pop_growth_rate'].max(), value=1.1)
    methane_emissions = st.slider('Annual methane emissions (in metric tons of CO₂ equivalents)', df['methane_emissions'].min(), df['methane_emissions'].max(), value=10.0*10**8)
    nitrous_oxide_emissions = st.slider('Annual nitrous oxide emissions (in metric tons of CO₂ equivalents)', int(df['nitrous_oxide_emissions'].min()), int(df['nitrous_oxide_emissions'].max()), value=10*10**7)
    gdp_usd = st.slider('GDP (in USD)', df['gdp_usd'].min(), df['gdp_usd'].max(), value=10.0*10**12)
    population = st.slider('Population', int(df['population'].min()), int(df['population'].max()), value=10*10**8)

    # make prediction
    input_data = pd.DataFrame({
        'area_code': area_code,
        'year': year,
        'co2_emissions_rate': co2_rate,
        'pop_growth_rate': pop_growth_rate,
        'methane_emissions': methane_emissions,
        'nitrous_oxide_emissions': nitrous_oxide_emissions,
        'gdp_usd': gdp_usd,
        'population': population
    }, index=[0])

    btn_display_input = st.button('display input data')
    if btn_display_input:
        st.write(input_data)

    st.divider()
    
    prediction = model.predict(input_data)[0].round(2)
    st.markdown(f"The predicted temperature change is: <span style='font-size:20px; color: #ff4b4b; font-weight: bold;'>{prediction} °C</span>", unsafe_allow_html=True)


elif page=="📌 Conclusion":
    # RF performance metrics
    rf_metrics = pd.read_excel('./input/RF_model_performance_metrics.xlsx')
    st.title('📌 Conclusion')
    st.subheader('After analyzing the data and developing ML models, we can conclude that:')
    st.markdown('''
* The temperature change is positively correlated with the CO₂ emissions rate, population growth rate, methane emissions, nitrous oxide emissions, and GDP.\n
    This means that as these factors increase, the temperature change also increases. This is because these factors contribute to the greenhouse effect, which traps heat in the atmosphere and causes the Earth to warm up.
* Continents such as Europe, Asia, and Africa experiences signifacnt impacts from global warming.
* The pattern of golbal warming impact may vary based on country's geographic position.
* The Random Forest Regressor model can predict the temperature change with reasonable accuracy based on the input features.''')
    st.dataframe(rf_metrics, use_container_width=True)
    st.markdown('''* For future work, we could explore more advanced ML/DL models, such as LSTM (Long Short-Term Memory), ARIMA (AutoRegressive Integrated Moving Average), SARIMA (Seasonal ARIMA), to improve the prediction accuracy and being able to predict the temperature change for different scenarios and time periods.''')

    st.subheader('Thank you for your attention!')
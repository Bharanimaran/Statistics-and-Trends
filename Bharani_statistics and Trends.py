import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def load_data(file_path):
    """ 
    Import information from a CSV file.
    
    Setting parameters:
        file_path (str): The CSV file path.
        
     Returns:
        DataFrame: Data loaded.
    """
    df = pd.read_csv(file_path, skiprows=4)
    return df

def process_data(df):
    """ 
    Transpose and choose pertinent columns to process the data.

    The original data is represented by the parameter df (DataFrame).

    Results include: - Tuple[DataFrame, DataFrame]: Transposed and processed data.
    
    """
    selected_columns = df[['Country Name'] + ['Indicator Name'] + [str(year) for year in range(1960, 2023)]]
    df_selected = selected_columns.copy()
    transposed_df = df_selected.transpose()
    transposed_df = transposed_df.reset_index()
    transposed_df.columns = transposed_df.iloc[0]
    transposed_df = transposed_df.drop(0)
    transposed_df = transposed_df.set_index('Country Name')
    transposed_df.index.name = 'Year'
    transposed_df.reset_index(inplace=True)

    return df_selected, transposed_df

file_path = r"C:\Users\Bharani\Downloads\New folder\API_19_DS2_en_csv_v2_6183479.csv"
df = load_data(file_path)
df_selected, transposed_df = process_data(df)

def describe_method(df_selected):
    """ 
    Produce fundamental statistics for columns with numbers.

    Setting parameters
        DataFrame: Selected by df_selected.

    Returns: DataFrame: Detailed statistical information.
    
    """
    skewness = df_selected.skew()
    kurt = df_selected.kurtosis()
    
    skew_kurt_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurt})
    
    describe_result = df_selected.describe().append(skew_kurt_df)
    
    return describe_result

def heat_map(df_selected):   
    indicators = ['Agricultural land (sq. km)', 'Cereal yield (kg per hectare)', 'Urban population', 'Population growth (annual %)','CO2 emissions from liquid fuel consumption (% of total)']
    selected_data = df_selected[df_selected['Indicator Name'].isin(indicators)]
    selected_years = ['2000']

    indicator_corr = selected_data.pivot_table(index='Country Name', columns='Indicator Name', values=selected_years)
    correlation_matrix = indicator_corr.corr()

    describe_result = df_selected.describe()


    # Plot correlation heatmap
    plt.figure(figsize=(7, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap between Indicators')
    plt.xlabel('Indicator Name')
    plt.ylabel('Indicator Name')
    plt.show()

heat_map(df_selected)

def line_plot(df, countries, years, indicator, plot_type='line'):
    """ 
     Make line plot for specific countries, years, and indicators.
     
     Setting parameters:
         df (DataFrame): DataFrame input.
         countries (list): A list of nations to be plotted.
         The list of years to be included in the plot is called "years."
         indicator (str): Plotting indicator name.
         plot_type (str): Plot type  line. 
     
    """
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")

    for country in countries:
        
        subset = df[(df['Country Name'] == country) & (df['Indicator Name'] == indicator)].drop(columns='Indicator Name')
        
  
        subset_melted = pd.melt(subset, id_vars=['Country Name'], var_name='Year', value_name='Value')
        
        
        subset_melted['Year'] = pd.to_numeric(subset_melted['Year'], errors='coerce')
        
    
        subset_filtered = subset_melted[subset_melted['Year'].isin(years)]
        
        if plot_type == 'line':
            sns.lineplot(x='Year', y='Value', data=subset_filtered, label=country, ci=None)
        elif plot_type == 'scatter':
            sns.scatterplot(x='Year', y='Value', data=subset_filtered, label=country)
    
    plt.title(f'{plot_type.capitalize()} Plot for Specific Countries and Years ({indicator})')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


countries_to_plot = ['India', 'Brazil', 'Australia']
years_to_plot = [2000, 2010, 2015]
indicator_to_plot = 'Cereal yield (kg per hectare)'
line_plot(df_selected, countries_to_plot, years_to_plot, indicator_to_plot, plot_type='line')


def bar_chart(df, countries, years, indicator):
    """
    Create a bar chart for specific countries, years, and an indicator.

    Parameters:
     df (DataFrame): Input DataFrame.
     countries (list): List of countries to include in the plot.
     years (list): List of years to include in the plot.
     indicator (str): Indicator name for plotting.
    """
    subset = df[(df['Country Name'].isin(countries)) & (df['Indicator Name'] == indicator)]
    subset = subset.melt(id_vars=['Country Name'], var_name='Year', value_name='Value')

    subset['Year'] = pd.to_numeric(subset['Year'], errors='coerce')

    subset = subset[subset['Year'].isin(years)]

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Country Name', y='Value', hue='Year', data=subset, ci=None)

    plt.title(f'Bar chart for specific years in {indicator}')
    plt.xlabel('Country')
    plt.ylabel('Value')

    # Format the year in legend without decimal points
    years_without_decimals = [int(year) for year in years]
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left', labels=[str(year) for year in years_without_decimals])

    plt.xticks(rotation=45, ha='right')
    plt.show()

countries_to_plot = ['India', 'Brazil', 'China', 'United States', 'Europe & Central Asia', 'South Africa']
years_to_plot = [2000, 2005, 2010, 2015]
indicator_to_plot = 'Urban population'

bar_chart(df_selected, countries_to_plot, years_to_plot, indicator_to_plot)




def histogram_plot(df, country, indicator):
    """
    Create a histogram for a specific indicator in a given country.

    Parameters:
     df (DataFrame): Input DataFrame.
     country (str): Name of the country for plotting.
     indicator (str): Indicator name for plotting.
    """
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    
    subset = df[(df['Country Name'] == country) & (df['Indicator Name'] == indicator)]

   
    values = subset.iloc[:, 2:].dropna(axis=1).values.flatten()

    
    sns.histplot(values, bins=20)

    plt.title(f'Histogram for {indicator} in {country}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend([country]) 
    plt.show()


country_to_plot = 'India'
indicator_to_plot = 'CO2 emissions from liquid fuel consumption (% of total)'

histogram_plot(df_selected, country_to_plot, indicator_to_plot)




def box_plot(df, countries, years, indicator):
    """
    Create a box plot for the specified indicator in multiple countries and years.

    Parameters:
     df (DataFrame): Input DataFrame.
     countries (list): List of country names to include in the plot.
     years (list): List of years to include in the plot.
     indicator (str): Indicator name for plotting.
    """
    subset = df[(df['Country Name'].isin(countries)) & (df['Indicator Name'] == indicator)]
    subset = subset.melt(id_vars=['Country Name'], var_name='Year', value_name='Value')

    subset['Year'] = pd.to_numeric(subset['Year'], errors='coerce')

    subset = subset[subset['Year'].isin(years)]

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Country Name', y='Value', data=subset)

    plt.title('Box plot for country with years in Population growth (annual %)')
    plt.xlabel('Country')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')
    # plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

countries_to_plot = ['India', 'Brazil', 'China', 'United States', 'Europe & Central Asia', 'South Africa']
years_to_plot = [2000, 2005, 2010, 2015]
indicator_to_plot = 'Population growth (annual %)'

box_plot(df_selected, countries_to_plot, years_to_plot, indicator_to_plot)



def pie_chart(df_selected, countries, year, indicator):
    
    """
    Create a pie chart for the specified indicator in multiple countries for a given year.

    Parameters:
     df_selected (DataFrame): Input DataFrame.
     countries (list): List of country names to include in the pie chart.
     year (int): The year for which the data is plotted.
     indicator (str): Indicator name for plotting.
    """

   
    subset = df_selected[(df_selected['Country Name'].isin(countries)) & (df_selected['Indicator Name'] == indicator)]
    subset = subset.melt(id_vars=['Country Name'], var_name='Year', value_name='Value')

   
    subset['Year'] = pd.to_numeric(subset['Year'], errors='coerce')

  
    subset = subset[subset['Year'] == year]

    
    plt.figure(figsize=(8, 8))
    explode=[0.01,0.01,0.01,0.01,0.01,0.01]
    wedges, texts, autotexts = plt.pie(subset['Value'], labels=subset['Country Name'], explode=explode, autopct='%1.1f%%', startangle=140)

    
    plt.legend(wedges, subset['Country Name'], title='Countries', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

   
    plt.setp(autotexts, size=10, weight="bold")

    plt.title(f'Pie Chart for {indicator} - Year {year}')
    plt.show()


countries_to_plot = ['India', 'Brazil', 'China', 'United States', 'Europe & Central Asia', 'South Africa']
year_to_plot = 2000
indicator_to_plot = 'Agricultural land (sq. km)'

pie_chart(df_selected, countries_to_plot, year_to_plot, indicator_to_plot)


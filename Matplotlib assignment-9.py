#Q1)Scipy:
#We have the min and max temperatures in a city In India for each months of the year. We would like to find a function to describe this and show it graphically, the dataset given below.
#Task:
#fitting it to the periodic function
#plot the fit

#Importing necessary libraries
import numpy as np                       
import pandas as pd                      
import matplotlib.pyplot as plt          
import scipy.optimize as scipy_optimize   

#Imput Data

Max=[39, 41, 43, 47, 49, 51, 45, 38, 37, 29, 27, 25]
Min=[21, 23, 27, 28, 32, 35, 31, 28, 21, 19, 17, 18]

# Convert temprature data into input data into numpy array
temp_max=np.array(Max)
temp_min=np.array(Min)

# Get year ( mothly) data
months= np.arange(12) #  month datra for a year

#sample plot

print("Sample plot for temprature distribution. \n")
plt.plot(months,temp_max,'ro',label='Max Temp')
plt.plot(months,temp_min,'bo',label='Min Temp')
plt.title("Tempratuire distribution of India in a Year")
plt.xlabel('Month')
plt.ylabel('Min and max temperature')
plt.legend()
plt.show()

# Create function
def yearly_temp_dist(times, avg, ampl, time_offset):
    return (avg + ampl * np.cos((times + time_offset) * 2 * np.pi / times.max()))


# Function execution and using scipy.optimize to fit data into curve 

popt_max, pcov_max = scipy_optimize.curve_fit(yearly_temp_dist, months,temp_max, [20, 10, 0])
popt_min, pcov_min = scipy_optimize.curve_fit(yearly_temp_dist, months,temp_min, [-40, 20, 0])

# Note Scipy.optimize.curve_fit returns 
#(1.) popt:an Optimal value array (1D array) & (2.) pcov: The estimated covariance of popt

print("Data for Max Temp\n")
print(popt_max, "\n")
print(pcov_max)
print("\nData for Min Temp\n")
print(popt_min,"\n")
print(pcov_min)


# generate days data to get temprature distribution throghout the year
days = np.linspace(0, 12, num=365)

#plot the data
plt.plot(months, temp_max, 'ro')
plt.plot(days, yearly_temp_dist(days, *popt_max),'r-', label='Max Temp')
plt.plot(months, temp_min, 'bo')
plt.plot(days, yearly_temp_dist(days, *popt_min),'b-', label='Min Temp')
plt.title("Tempratuire distribution of India in a Year")
plt.xlabel('Month')
plt.ylabel('Temperature ($^\circ$C)')
plt.legend()
plt.show()


#Q2)Matplotlib: This assignment is for visualization using matplotlib: data to use:

#titanic = pd.read_csv(url)

#Charts to plot:
#Create a pie chart presenting the male/female proportion
#Create a scatterplot with the Fare paid and the Age, differ the plot color by gender

#Load Titanic data 
titanic=pd.read_csv('https://raw.githubusercontent.com/Geoyi/Cleaning-Titanic-Data/master/titanic_original.csv')
titanic.head()

#Create a pie chart presenting the male/female proportion

# Get the ratio of Male and female
ratio_male_female= titanic['sex'].value_counts( normalize = True )
#Pie chart for Male-Female ratio representation
labels=['Male','Female']
colors=['xkcd:azure','xkcd:yellowgreen']

patches, text , percentage =plt.pie(ratio_male_female,  labels=labels, colors=colors,autopct='%1.2f%%',shadow=True)
#increase the size of text of pie chart and percentage
text[0].set_fontsize(12)
percentage[0].set_fontsize(12)
text[1].set_fontsize(12)
percentage[1].set_fontsize(12)

plt.legend(patches, labels, loc=1)
plt.tight_layout()
plt.title('"Ratio of male/female"').set_fontsize(14)
plt.show()

#Create a scatterplot with the Fare paid and the Age, differ the plot color by gender

#Get data of sex, age , fare from titanic dataset

titanic_scatter=titanic[['sex','fare','age']].dropna( how = "all" ) #  remove NA /NaN if vailable in all columns


# identify NA / NaN values in for fare  and age columns
import numpy as np
index_fare = titanic_scatter['fare'].index[titanic_scatter['fare'].apply(np.isnan)]
index_age = titanic_scatter['age'].index[titanic_scatter['age'].apply(np.isnan)]
titanic_scatter.iloc[index_age]
titanic_scatter.iloc[index_fare]


# fill NA/ NaN with Zero (0.0) for fare  and age columns

titanic_scatter[['fare', 'age']] = titanic_scatter[['fare', 'age']].fillna(value=0)

# Generate Plot 

mapping = {'male' : 'blue', 'female' : 'red'}

# Import patvh and pyplot module for Legend generation
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

red_patch = mpatches.Patch(color='blue', label=list(mapping.keys())[0])
blue_patch = mpatches.Patch(color='red', label=list(mapping.keys())[1])


plt.scatter(titanic_scatter['age'],  titanic_scatter['fare'], alpha=0.5, c=titanic_scatter['sex'].map(mapping), label=mapping)

plt.title('Scatter-plot for Fare with Age based upon Gender').set_fontsize(12)
plt.xlabel('Age').set_fontsize(12)
plt.ylabel('Fare').set_fontsize(12)
plt.legend(handles=[red_patch,blue_patch] , loc=0)
plt.show()












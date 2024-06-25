#!/usr/bin/env python
# coding: utf-8

# ## Importing Required Libraries
# 
# This cell imports necessary libraries for data manipulation, statistical modeling, visualization, and plotting.
# 

# In[1]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
from itertools import product
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.stats.multicomp import MultiComparison

# Ensure all necessary libraries are imported for the subsequent analysis.


# ## Simulating Example Data
# 
# This cell generates example data simulating a study with factors like Diet, Housing, and Season, along with a response variable WeightGain.
# 

# In[2]:


np.random.seed(123)

# Define levels for Diet, Housing, and Season
diet_levels = ["Diet1", "Diet2", "Diet3"]
housing_levels = ["Housing1", "Housing2", "Housing3"]
season_levels = ["Winter", "Summer", "Spring"]

# Create a DataFrame with simulated data
# Sample data
data = pd.DataFrame({
    'Diet': np.random.choice(['Diet1', 'Diet2', 'Diet3'], 100),
    'Housing': np.random.choice(['Housing1', 'Housing2', 'Housing3'], 100),
    'Season': np.random.choice(['Winter', 'Spring', 'Summer'], 100),
    'WeightGain': np.random.normal(5, 2, 100)
})

data.head()


# ## Fitting Linear Model
# 
# This cell fits a linear regression model to the simulated data to analyze the effects of Diet, Housing, and Season on WeightGain.
# 

# In[3]:


# Fit linear model
model = ols('WeightGain ~ Diet * Housing * Season', data=data).fit()

# Display summary of the model
print(model.summary())


# ## Creating Interaction Plot
# 
# This cell creates an interaction plot using Seaborn to visualize the effects of Diet and Housing on WeightGain.
# 

# In[4]:


# Interaction plot using Seaborn
plt.figure(figsize=(10, 6))
sns.pointplot(x='Diet', y='WeightGain', hue='Housing', data=data, ci=None)
plt.title('Interaction Plot of Diet and Housing on Weight Gain')
plt.xlabel('Diet')
plt.ylabel('Mean Weight Gain')
plt.legend(title='Housing')
plt.show()


# ## Performing Tukey's HSD Test
# 
# This cell perform Tukey's Honestly Significant Difference (HSD) test across all factors (Diet, Housing, Season, and their interactions), we'll need to conduct pairwise comparisons for each factor and their combinations.
# 

# In[5]:


# Define factors and their levels
factors = ['Diet', 'Housing', 'Season']
factor_levels = {
    'Diet': diet_levels,
    'Housing': housing_levels,
    'Season': season_levels
}


# In[6]:


# Perform Tukey's HSD test for each factor and their interactions
for factor in factors:
    mc = MultiComparison(data['WeightGain'], data[factor])
    result = mc.tukeyhsd()
    print(f"Tukey's HSD Test for {factor}:")
    print(result)
    print("\n")


# In[7]:


# Perform Tukey's HSD test for interactions
for i in range(len(factors)):
    for j in range(i+1, len(factors)):
        factor1 = factors[i]
        factor2 = factors[j]
        mc = MultiComparison(data['WeightGain'], data.groupby([factor1, factor2]).grouper.group_info[0])
        result = mc.tukeyhsd()
        print(f"Tukey's HSD Test for Interaction between {factor1} and {factor2}:")
        print(result)
        print("\n")


# In[8]:


# Perform Tukey's HSD test for interactions between Diet, Housing, and Season
mc = MultiComparison(data['WeightGain'], data.groupby(factors).grouper.group_info[0])
result = mc.tukeyhsd()
print(f"Tukey's HSD Test for Interaction between Diet, Housing, and Season:")
print(result)


# In[9]:


# Generate all combinations of factor levels
factor_combinations = list(product(*factor_levels.values()))

# Perform Tukey's HSD test for the three-way interaction
mc = MultiComparison(data['WeightGain'], data.groupby(factors).grouper.group_info[0])
result = mc.tukeyhsd()

print(f"Tukey's HSD Test for Interaction between Diet, Housing, and Season (Three-way Interaction):")
print(result)


# To visualize the interaction among the three factors (Diet, Housing, and Season), you can create a plot that shows how the mean WeightGain varies across different combinations of these factors. One effective way to visualize such interactions is by using a 3D plot or a faceted plot (also known as small multiples). 
# 

# In[10]:


# Using Faceted Plot (seaborn):
sns.set(style="whitegrid")
g = sns.catplot(x='Diet', y='WeightGain', hue='Season', col='Housing', data=data, kind='point', aspect=1.2)
g.set_axis_labels("Diet", "Mean Weight Gain")
g.set_titles("Housing: {col_name}")
plt.subplots_adjust(top=0.85)
g.fig.suptitle('Interaction Plot of Diet, Housing, and Season on Weight Gain', fontsize=16)
plt.show()


# ## Creating 3D Surface Plot using Plotly
# 
# This cell creates a 3D surface plot to visualize the predicted WeightGain based on Diet, Housing, and Season.
# 

# In[11]:


# Extract unique levels of Season from the original data
season_levels = data['Season'].unique()


# In[12]:


# Create a grid for the surface plot
diet_levels = ['Diet1', 'Diet2', 'Diet3']
housing_levels = ['Housing1', 'Housing2', 'Housing3']


# In[13]:


# Create the grid for predictions
grid = pd.DataFrame({
    'Diet': np.repeat(diet_levels, len(housing_levels) * len(season_levels)),
    'Housing': np.tile(np.repeat(housing_levels, len(season_levels)), len(diet_levels)),
    'Season': np.tile(season_levels, len(diet_levels) * len(housing_levels))
})


# In[14]:


# Ensure the grid levels match those used in the model
grid['Diet'] = pd.Categorical(grid['Diet'], categories=diet_levels)
grid['Housing'] = pd.Categorical(grid['Housing'], categories=housing_levels)
grid['Season'] = pd.Categorical(grid['Season'], categories=season_levels)


# In[15]:


# Predict weight gain using the fitted model
grid['PredictedWeightGain'] = model.predict(grid)


# In[16]:


# Map categorical variables to numeric for plotting
diet_numeric = {level: i for i, level in enumerate(diet_levels)}
housing_numeric = {level: i for i, level in enumerate(housing_levels)}
season_numeric = {level: i for i, level in enumerate(season_levels)}


# In[17]:


# Map the data for plotting
grid['Diet_num'] = grid['Diet'].map(diet_numeric)
grid['Housing_num'] = grid['Housing'].map(housing_numeric)


# In[18]:


# Create the surface plot
surface_plot = go.Surface(
    x=grid['Diet_num'].values.reshape(len(diet_levels), len(housing_levels) * len(season_levels)),
    y=grid['Housing_num'].values.reshape(len(diet_levels), len(housing_levels) * len(season_levels)),
    z=grid['PredictedWeightGain'].values.reshape(len(diet_levels), len(housing_levels) * len(season_levels)),
    colorscale='Viridis'
)


# In[19]:


# Scatter plot
season_colors = {'Winter': 'blue', 'Summer': 'red', 'Spring': 'green'}
data['SeasonColor'] = data['Season'].map(season_colors)
scatter_plot = go.Scatter3d(
    x=data['Diet'].map(diet_numeric),
    y=data['Housing'].map(housing_numeric),
    z=data['WeightGain'],
    mode='markers',
    marker=dict(size=5, color=data['SeasonColor'])
)


# In[20]:


# Create and display the surface plot separately
fig_surface = go.Figure(surface_plot)
fig_surface.update_layout(
    title='3D Surface Plot of Diet, Housing, and Weight Gain',
    scene=dict(
        xaxis=dict(title='Diet', tickvals=list(diet_numeric.values()), ticktext=list(diet_numeric.keys()), tickangle=-45),
        yaxis=dict(title='Housing', tickvals=list(housing_numeric.values()), ticktext=list(housing_numeric.keys()), tickangle=-45),
        zaxis=dict(title='Weight Gain'),
        aspectmode='cube'
    ),
    height=600, width=1200, margin=dict(l=50, r=50, b=50, t=100)
)

# Tick Angle: tickangle=-45 rotates the tick labels on the x and y axes by 45 degrees, which can help in preventing overlap and making them more readable.
# Margin Adjustment: margin=dict(l=50, r=50, b=50, t=100) adjusts the left, right, bottom, and top margins of the plot area to ensure sufficient space for axis labels and titles.
# Adjust the margin values further if needed based on your specific plot dimensions and requirements.

# Save the figure as an HTML file
fig_surface.write_html("3D_surface_plot.html")

# Display the figure
fig_surface.show()


# In[21]:


# Create and display the scatter plot separately
fig_scatter = go.Figure(scatter_plot)
fig_scatter.update_layout(
    title='Scatter Plot of Diet, Housing, and Weight Gain',
    scene=dict(
        xaxis=dict(title='Diet', tickvals=list(diet_numeric.values()), ticktext=list(diet_numeric.keys())),
        yaxis=dict(title='Housing', tickvals=list(housing_numeric.values()), ticktext=list(housing_numeric.keys())),
        zaxis=dict(title='Weight Gain')
    ),
    height=600, width=800
)

# Save the figure as an HTML file
fig_scatter.write_html("3D_scatter_plot.html")

# Display the figure
fig_scatter.show()


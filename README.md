# Exploring Factors Influencing Weight Gain: Data Simulation, Regression Analysis, and 3D Visualization

### Author: Hamdy Abdel-Shafy
### Date: June 2024
### Affiliation: Department of Animal Production, Cairo University, Faculty of Agriculture


## Overview

This repository contains Python code that simulates a study investigating how factors like Diet, Housing, and Season influence Weight Gain.
It performs statistical analysis using ANOVA (Analysis of Variance) to examine the effects of these factors on weight gain. 
Additionally, Tukey's HSD (Honestly Significant Difference) test is applied for pairwise comparisons between groups.

## Files

- **facorial_3by3_ANOVA_Tukey_3d.html**: Python code through html containing the entire analysis pipeline.
# To see the content of this file, you need to doenload it locally and open it with any web browser
- **readMe_python.md**: This file, providing an overview and explanation of the code.

---
Here’s a breakdown of what each section of the code does:
---

## 1. Importing Required Libraries

This section imports necessary libraries for data manipulation, statistical modeling, and visualization. These libraries include:
- `pandas`: for data manipulation using DataFrames.
- `numpy`: for numerical operations and generating random data.
- `statsmodels.api`: for statistical models like linear regression.
- `seaborn` and `matplotlib.pyplot`: for data visualization and plotting.
- `plotly`: for interactive 3D visualization.
- `statsmodels.stats.multicomp.MultiComparison`: for conducting Tukey's HSD test.

## 2. Simulating Example Data

This section generates example data mimicking a study with three factors: Diet, Housing, and Season, along with a response variable WeightGain.
It creates a DataFrame (`data`) containing:
- Categorical variables (`Diet`, `Housing`, `Season`).
- Continuous variable (`WeightGain`) simulated using normal distribution.

## 3. Fitting Linear Model

This part fits a linear regression model to analyze how Diet, Housing, and Season collectively affect WeightGain. 
The model is fitted using `statsmodels`' `ols` (ordinary least squares) method.

## 4. Creating Interaction Plot

Uses `seaborn` to create an interaction plot showing the effects of Diet and Housing on WeightGain. 
This plot visualizes how the mean WeightGain varies across different levels of Diet and Housing.

## 5. Performing Tukey's HSD Test

Conducts Tukey's Honestly Significant Difference (HSD) test to identify significant differences in WeightGain across different levels of each factor (Diet, Housing, Season) and their interactions. 
The results are printed for each factor and their combinations.

## 6. Visualizing Three-Way Interaction

Demonstrates how to visualize interactions among Diet, Housing, and Season:
- Uses a faceted plot (`seaborn.catplot`) to show interactions of Diet and Season across different Housing levels.
- Creates a 3D surface plot (`plotly`) to visualize predicted WeightGain based on combinations of Diet, Housing, and Season.

## 7. Creating 3D Surface Plot using Plotly

This section demonstrates how to create a 3D surface plot to visualize the predicted WeightGain based on combinations of Diet, Housing, and Season:
- The data is organized into a grid (`grid`) representing all possible combinations of Diet, Housing, and Season.
- Predictions are made using the previously fitted linear regression model (`model.predict`).
- `plotly` is used to generate an interactive 3D surface plot (`go.Surface`) showing how WeightGain varies across different combinations of Diet and Housing, colored by Season.

## 8. Creating Scatter Plot

This part creates a scatter plot (`go.Scatter3d`) to visualize actual WeightGain data points across different levels of Diet and Housing, colored by Season. 
This plot provides a complementary view to the 3D surface plot, showing individual data points and their distributions.

---

### Visualization and Interpretation

- **Surface Plot**: Use the interactive 3D surface plot to explore how predicted WeightGain changes with different combinations of Diet, Housing, and Season. 
					Adjust the view angles and interact with the plot to gain insights into the relationships between these factors and WeightGain.
  
- **Scatter Plot**: Analyze the actual WeightGain data points to understand their distribution and overlap across different levels of Diet, Housing, and Season. 
					This helps in validating the trends observed in the surface plot.

---

### How to Interpret the Results

- **Interpretation of Interaction**: Look for patterns in the plots to understand how combinations of Diet, Housing, and Season collectively impact WeightGain. 
									Pay attention to any significant differences or trends observed across different factor levels.

- **Further Analysis**: Consider conducting additional statistical tests or exploring more advanced modeling techniques based on the insights gained from these visualizations. 
						This could involve deeper analysis of interactions or exploring non-linear relationships if necessary.

---

### Conclusion

This repository provides a comprehensive example of using Python for data simulation, linear regression modeling, and advanced visualization techniques. 
Beginners can use this as a starting point to learn about data analysis and visualization, while also exploring how different factors interact in influencing a response variable like WeightGain.

---

### How to Use This Repository

1. **Clone the Repository**: Clone this repository to your local machine using Git.
   
   ```
   git clone <repository_url>
   ```

2. **Install Dependencies**: Ensure you have Python installed along with the necessary libraries (`pandas`, `numpy`, `statsmodels`, `seaborn`, `matplotlib`, `plotly`).

3. **Run the Notebook**: Execute the Jupyter notebook (`facorial_3by3_ANOVA_Tukey_3d.ipynb`) to see the code in action. 
						Follow each section to understand how data is generated, models are fitted, and visualizations are created.

4. **Explore and Modify**: Feel free to modify the code or adapt it to your own datasets. 
						   Experiment with different variables, models, or visualization techniques.

## License

This project is licensed under the [MIT License](LICENSE).

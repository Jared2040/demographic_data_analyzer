import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import the data from 'medical_examination.csv' and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2. Add an 'overweight' column to the data. BMI calculation: weight / height^2
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)
df.drop('BMI', axis=1, inplace=True)

# 3. Normalize the cholesterol and glucose columns: if the value is 1 (normal), set it to 0; if more than 1, set it to 1.
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4. Draw the Categorical Plot in the draw_cat_plot function.
def draw_cat_plot():
    # 5. Create a DataFrame for the cat plot using pd.melt with the values from cholesterol, gluc, smoke, alco, active, overweight.
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group and reformat the data in df_cat to split by cardio and show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat.rename(columns={'size': 'total'}, inplace=True)

    # 7. Create the catplot using seaborn.
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar', height=5, aspect=1
    ).fig

    # 8. Get the figure for the output.
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar', height=5, aspect=1
    ).fig

    # 9. Save the plot to a file.
    fig.savefig('catplot.png')
    return fig

# 10. Draw the Heat Map in the draw_heat_map function.
def draw_heat_map():
    # 11. Clean the data in df_heat by filtering out incorrect patient segments.
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic <= Systolic
        (df['height'] >= df['height'].quantile(0.025)) &  # Height >= 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # Height <= 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight >= 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # Weight <= 97.5th percentile
    ]

    # 12. Calculate the correlation matrix.
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle of the correlation matrix.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure.
    fig, ax = plt.subplots(figsize=(10, 10))

    # 15. Plot the heatmap.
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.1f', square=True,
        cmap='coolwarm', cbar_kws={'shrink': 0.5}
    )
    # 16. Save the plot to a file.
    fig.savefig('heatmap.png')
    return fig

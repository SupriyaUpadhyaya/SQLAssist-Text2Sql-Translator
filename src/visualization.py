import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

excel_files = ['Aggregation.xlsx', 'Basic SQL.xlsx', 'CTEs.xlsx', 'Multiple_joins.xlsx',
               'Set operations.xlsx', 'Single join.xlsx', 'Subqueries.xlsx', 'Window functions.xlsx']

# Iterate through each Excel file
for file in excel_files:
    df = pd.read_excel(file)
    min_bleu = df['Bleu Score'].min()
    max_bleu = df['Bleu Score'].max()
    min_rouge = df['Rouge Score'].min()
    max_rouge = df['Rouge Score'].max()

    print(f"File: {file}")
    print("Minimum Bleu Score:", min_bleu)
    print("Maximum Bleu Score:", max_bleu)
    print("Minimum Rouge Score:", min_rouge)
    print("Maximum Rouge Score:", max_rouge)
    print()

axes = axes.flatten()
dfs = []
for i, file in enumerate(excel_files):
    print(file)
    data = pd.read_excel(file)
    dfs.append(data)
    #axes[i].boxplot(df['Bleu Score'])
    #axes[i].set_title(f'Bleu Score {file}')
    #axes[i].set_xticks([1])
    #axes[i].set_xticklabels(['Bleu Score'])
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.head()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Use sns.boxplot() to create the box plot
#sns.boxplot(x='sql_complexity', y='bleu_score', data=merged_df)
#sns.boxplot(data=titanic, x="class", y="age", hue="alive")
plt.figure(figsize=(11,6))
# create 3rd grouped boxplot
sns.boxplot(x = merged_df['sql_complexity'],
			y = merged_df['Bleu Score'],
			hue = merged_df['sql_complexity'],
			palette = 'husl')


# Add labels and title
plt.xlabel('SQL Complexity')
plt.ylabel('Bleu Score')
plt.title('Box Plot of Bleu Score for Each SQL Complexity - For gretelai/synthetic_text_to_sql')

len(merged_df)

selected_df = merged_df[['Bleu Score', 'Rouge Score']]

# Create box plot using seaborn
sns.boxplot(data=selected_df)

# Add labels and title
plt.xlabel('Score Type')
plt.ylabel('Score')
plt.title('Box Plot of BLEU Score and ROGUE Score - For gretelai/synthetic_text_to_sql')

# Show the plot
plt.show()
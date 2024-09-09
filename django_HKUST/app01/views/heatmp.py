import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data from the image
data = {
    'Retrieval': [6, 2, 9],
    'Summary': [11, 15, 3],
    'Recommendation': [12, 18, 4],
    'Trend': [11, 5, 4]
}

# Index (rows)
index = ['Student', 'Scholar', 'Admin']

# Creating a DataFrame
df = pd.DataFrame(data, index=index)

# Create a heatmap
plt.figure(figsize=(10, 5))
sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=False, fmt='g', vmin=0, vmax=25, square=False)

# Adding labels and title
plt.title("Number of Questions(Category)", fontsize=20, fontweight='bold')
# plt.xlabel("Features", fontsize=12)
# plt.ylabel("User Types", fontsize=12)

# Show the heatmap
plt.show()


# New data from the image
data_new = {
    'Department': [8, 10, 20],
    'Author': [13, 6, 0],
    'Year': [9, 11, 5],
    'Venue': [3, 10, 2],
    'Keyword': [21, 22, 5]
}

# Index (rows)
index_new = ['Student', 'Scholar', 'Admin']

# Creating a DataFrame
df_new = pd.DataFrame(data_new, index=index_new)

# Create a heatmap
plt.figure(figsize=(10, 5))
# sns.heatmap(df_new, annot=True, cmap="YlGnBu", cbar=False, fmt='g')

# 自定义颜色范围，设置vmin和vmax
sns.heatmap(df_new, annot=True, cmap="YlGnBu", cbar=True, fmt='g', vmin=0, vmax=25, square=False)

# Adding labels and title
plt.title("Number of Questions(Faceted)", fontsize=20, fontweight='bold')
# plt.xlabel("Attributes", fontsize=12)
# plt.ylabel("User Types", fontsize=12)

# Show the heatmap
plt.show()







# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
#
# # Data for the first heatmap
# data = {
#     'Retrieval': [6, 2, 9],
#     'Summary': [11, 15, 3],
#     'Recommendation': [12, 18, 4],
#     'Trend': [11, 5, 4]
# }
#
# # Index (rows)
# index = ['Student', 'Scholar', 'Admin']
#
# # Creating a DataFrame for the first heatmap
# df = pd.DataFrame(data, index=index)
#
# # New data for the second heatmap
# data_new = {
#     'Department': [8, 10, 20],
#     'Author': [13, 6, 0],
#     'Year': [9, 11, 5],
#     'Venue': [3, 10, 2],
#     'Keyword': [21, 22, 5]
# }
#
# # Index (rows) for the second heatmap
# index_new = ['Student', 'Scholar', 'Admin']
#
# # Creating a DataFrame for the second heatmap
# df_new = pd.DataFrame(data_new, index=index_new)
#
# # Create a figure with 1 row and 2 columns of subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Adjust figsize to control the overall size
#
# # First heatmap on the left
# sns.heatmap(df, annot=True, cmap="YlGnBu", cbar=True, fmt='g', vmin=0, vmax=25, ax=ax1)
# ax1.set_title("Number of Questions (Category)", fontsize=14)
#
# # Second heatmap on the right without color bar
# sns.heatmap(df_new, annot=True, cmap="YlGnBu", cbar=False, fmt='g', vmin=0, vmax=25, ax=ax2)
# ax2.set_title("Number of Questions (Faceted)", fontsize=14)
#
# # Add a single color bar to the right of the second heatmap
# cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # Adjust position and size of the color bar
# norm = plt.Normalize(vmin=0, vmax=25)
# sm = plt.cm.ScalarMappable(cmap="YlGnBu", norm=norm)
# cbar = fig.colorbar(sm, cax=cbar_ax)
# cbar.set_label('Number of Questions', fontsize=12)
#
# # Show the combined heatmap
# plt.tight_layout()  # Adjust layout to ensure no overlap
# plt.show()
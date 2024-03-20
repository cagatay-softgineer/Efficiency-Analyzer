import pandas as pd

# Load the CSV file
df = pd.read_csv('CSV/Merged.csv')
### 0 - 7724 - 10536 - 12474 - df.last_valid_index() + 1 
# Define the range of row indices you want to keep

index_list = [0, 7724, 10536, 12474, df.last_valid_index() + 1]
index_list = [0, 1786, 8354, 16620, df.last_valid_index() + 1]
cut_df_list = [df.iloc[index_list[i]:index_list[i+1]] for i in range(len(index_list)-1)]

# Now cut_df_list contains the cut parts of your dataframe
# For example:
part_1 = cut_df_list[0]
part_2 = cut_df_list[1]
part_3 = cut_df_list[2]
part_4 = cut_df_list[3]

for i, part_df in enumerate(cut_df_list, start=1):
    part_df.to_csv(f'CSV/Cut{i}.csv', index=False)

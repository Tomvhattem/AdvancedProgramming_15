from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

file_name = 'tested_molecules-1.csv'

def read_data(file_name):
    df = pd.read_csv(file_name)
    return df

df_molecules = read_data(file_name)

df_molecules = pd.read_csv(file_name)

descriptions_list = [n[0] for n in Descriptors._descList]
descriptions_calculator =  MoleculeDescriptors.MolecularDescriptorCalculator(descriptions_list)

def get_mols(df):
    mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    return mols
mols = get_mols(df_molecules)

def calculate_descriptions(description_calculator):
    calculated_descriptions = [description_calculator.CalcDescriptors(m) for m in mols]
    return calculated_descriptions
calculated_descriptions = calculate_descriptions(descriptions_calculator)

physical_descriptions = [i for i in descriptions_list if not i.startswith('fr_')]
group_descriptions = [i for i in descriptions_list if i.startswith('fr_')]

pysc_calc = MoleculeDescriptors.MolecularDescriptorCalculator(physical_descriptions)
group_calc = MoleculeDescriptors.MolecularDescriptorCalculator(group_descriptions)

calculated_physical_descriptions = calculate_descriptions(pysc_calc)
calculated_group_descriptions = calculate_descriptions(group_calc)

df_physical = pd.DataFrame(calculated_physical_descriptions.copy())  # Create a copy to avoid modifying the original DataFrame
df_group = pd.DataFrame(calculated_group_descriptions.copy())

scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df_physical), columns=df_physical.columns)
df_normalized.columns = physical_descriptions
df_group.columns = group_descriptions

df_group = pd.get_dummies(df_group)


df_molecules.set_index('SMILES',inplace=True)
df_group.index = df_molecules.index
df_normalized.index = df_molecules.index
df_data = pd.concat([df_molecules,df_normalized,df_group],axis=1)

df_data.corr()

from sklearn.decomposition import PCA

df_data.set_index('ALDH1_inhibition', inplace=True)

# Filter dataframe based on ALDH1_inhibition values
df_filtered = df_data[df_data.index.isin([0, 1])]


pca = PCA()
X_pca = pca.fit_transform(df_filtered)

# (ii.) get basic info
n_components = len(pca.explained_variance_ratio_)
explained_variance = pca.explained_variance_ratio_
cum_explained_variance = np.cumsum(explained_variance)
idx = np.arange(n_components)+1
df_explained_variance = pd.DataFrame([explained_variance, cum_explained_variance], 
                                     index=['explained variance', 'cumulative'], 
                                     columns=idx).T
mean_explained_variance = df_explained_variance.iloc[:,0].mean() # calculate mean explained variance
# (iii.) Print explained variance as plain text
print('PCA Overview')
print('='*40)
print("Total: {} components".format(n_components))
print('-'*40)
print('Mean explained variance:', round(mean_explained_variance,3))
print('-'*40)
print(df_explained_variance.head(20))
print('-'*40)

#limit plot to x PC
limit_df = 20 #Changeble 

df_explained_variance_limited = df_explained_variance.iloc[:limit_df,:]
#make scree plot
fig, ax1 = plt.subplots(figsize=(15,6))
ax1.set_title('Explained variance across principal components', fontsize=14)
ax1.set_xlabel('Principal component', fontsize=12)
ax1.set_ylabel('Explained variance', fontsize=12)
ax2 = sns.barplot(x=idx[:limit_df], y='explained variance', data=df_explained_variance_limited, palette='summer')
ax2 = ax1.twinx()
ax2.grid(False)
ax2.set_ylabel('Cumulative', fontsize=14)
ax2 = sns.lineplot(x=idx[:limit_df]-1, y='cumulative', data=df_explained_variance_limited, color='#fc8d59')
ax1.axhline(mean_explained_variance, ls='--', color='#fc8d59') #plot mean
ax1.text(-.8, mean_explained_variance+(mean_explained_variance*.05), "average", color='#fc8d59', fontsize=14) #label y axis
max_y1 = max(df_explained_variance_limited.iloc[:,0])
max_y2 = max(df_explained_variance_limited.iloc[:,1])
ax1.set(ylim=(0, max_y1+max_y1*.1))
ax2.set(ylim=(0, max_y2+max_y2*.1))
plt.show()

# Access variable loadings
variable_loadings = pca.components_

# Create a DataFrame to display the loadings
df_loadings = pd.DataFrame(variable_loadings, columns=df_filtered.columns)

# Count variable occurrences across the first 10 components
num_components = 10
num_variables = 100

variable_counts = df_loadings.iloc[:num_components].abs().sum().sort_values(ascending=False).head(num_variables)

# Display the most used variables
print("Top {} Most Used Variables:".format(num_variables))
for i in range(num_variables):
    print(f"{i+1:2d}. {variable_counts.index[i]:<25s} {variable_counts[i]:.3f}")
import sys
import pandas as pd
import requests
from rdkit import Chem
import numpy as np
from rdkit.Chem import Descriptors
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

try:
    r = requests.get('https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv')
    if(r.status_code == 200):
        sol = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney.csv')
except Exception as e:
    print (e)

mol_list= []
for element in sol.SMILES:
  mol = Chem.MolFromSmiles(element)
  mol_list.append(mol)

def generate(smiles):
    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem) 
        moldata.append(mol)
       
    baseData= np.arange(1,1)
    i=0  
    for mol in moldata:        
        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
      
        row = np.array([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds])   
    
        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1      
    
    columnNames=["MolLogP","MolWt","NumRotatableBonds"]   
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)
    
    return descriptors

df = generate(sol.SMILES)

def AromaticAtoms(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  sum_aa_count = sum(aa_count)
  return sum_aa_count

desc_AromaticProportion = [AromaticAtoms(element)/Descriptors.HeavyAtomCount(element) for element in mol_list]
df_desc_AromaticProportion = pd.DataFrame(desc_AromaticProportion, columns=['AromaticProportion'])

X = pd.concat([df,df_desc_AromaticProportion], axis=1)
Y = sol.iloc[:,1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

def training_plot():
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    print('Coefficients:', model.coef_)
    print('Intercept:', model.intercept_)
    print('Mean squared error (MSE): %.2f'% mean_squared_error(Y_train, Y_pred_train))
    print('Coefficient of determination (R^2): %.2f'% r2_score(Y_train, Y_pred_train))

    Y_pred_test = model.predict(X_test)
    print('Coefficients:', model.coef_)
    print('Intercept:', model.intercept_)
    print('Mean squared error (MSE): %.2f'% mean_squared_error(Y_test, Y_pred_test))
    print('Coefficient of determination (R^2): %.2f'% r2_score(Y_test, Y_pred_test))

    full = linear_model.LinearRegression()
    full.fit(X, Y)
    full_pred = model.predict(X)

    print('Coefficients:', full.coef_)
    print('Intercept:', full.intercept_)
    print('Mean squared error (MSE): %.2f'% mean_squared_error(Y, full_pred))
    print('Coefficient of determination (R^2): %.2f'% r2_score(Y, full_pred))

    full_yintercept = '%.2f' % full.intercept_
    full_LogP = '%.2f LogP' % full.coef_[0]
    full_MW = '%.4f MW' % full.coef_[1]
    full_RB = '+ %.4f RB' % full.coef_[2]
    full_AP = '%.2f AP' % full.coef_[3]

    print('LogS = ' + ' ' + full_yintercept + ' ' + full_LogP + ' ' + full_MW + ' ' + full_RB + ' ' + full_AP)

    plt.figure(figsize=(11,5))

    # 1 row, 2 column, plot 1
    plt.subplot(1, 2, 1)
    plt.scatter(x=Y_train, y=Y_pred_train, c="#7CAE00", alpha=0.3)

    z = np.polyfit(Y_train, Y_pred_train, 1)
    p = np.poly1d(z)
    plt.plot(Y_test,p(Y_test),"#F8766D")

    plt.ylabel('Predicted LogS')
    plt.xlabel('Experimental LogS')

    # 1 row, 2 column, plot 2
    plt.subplot(1, 2, 2)
    plt.scatter(x=Y_test, y=Y_pred_test, c="#619CFF", alpha=0.3)

    z = np.polyfit(Y_test, Y_pred_test, 1)
    p = np.poly1d(z)
    plt.plot(Y_test,p(Y_test),"#F8766D")

    plt.xlabel('Experimental LogS')

    plt.savefig('plot_horizontal_logS.png')
    plt.show()
import pandas as pd
from pandas.core.frame import DataFrame
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools



def conjbonds(dataframe: DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['Conjugated Bonds']=0
    for i in range(len(dataframe)):
        n=0
        m=dataframe.loc[i, 'temp']
        for j in range(m.GetNumBonds()):
            if m.GetBondWithIdx(j).GetIsConjugated() is True:
                n=n+1
        dataframe.loc[i, 'Conjugated Bonds']=n
    dataframe=dataframe.drop('temp', axis = 1)
    return dataframe

def amide(dataframe:DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    lactam = Chem.MolFromSmarts('nc=O')
    amide = Chem.MolFromSmarts('NC=O')
    dataframe['Lactam'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(lactam)))
    dataframe['Amide'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(amide)))
    dataframe['N-C=O'] = dataframe['Lactam'] + dataframe['Amide']
    dataframe = dataframe.drop(['temp','Amide', 'Lactam'], axis=1)
    return dataframe

def carbonyl(dataframe:DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    carbonyl = Chem.MolFromSmarts('[C,c]=O')
    dataframe['C=O'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(carbonyl)))
    dataframe = dataframe.drop(['temp'], axis = 1)
    return dataframe

def PiDPiA(dataframe:DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    PiDPiA = Chem.MolFromSmarts('[SX2,sX2,OX2,oX2,NX3,nX3]-,:[N,n;X2]')
    dataframe['PiD-PiA'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(PiDPiA)))
    dataframe = dataframe.drop(['temp'], axis=1)
    return dataframe
            
def AP(dataframe:DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['AA'] = dataframe['temp'].apply(lambda m: len(m.GetAromaticAtoms()))
    dataframe['HA'] = dataframe['temp'].apply(lambda m: Descriptors.HeavyAtomCount(m))
    dataframe['AP'] = dataframe['AA']/dataframe['HA']
    dataframe = dataframe.drop(['temp', 'AA', 'HA'], axis = 1)
    return dataframe

def ARR(dataframe:DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    for i in range(len(dataframe)):
        m = dataframe.loc[i, 'temp']
        ab=0
        for j in range(m.GetNumBonds()):
            if m.GetBondWithIdx(j).GetIsAromatic() is True:
                ab+=1
        dataframe.loc[i, 'AB'] = ab
    dataframe['TB'] = dataframe['temp'].apply(lambda m: m.GetNumBonds())
    dataframe['ARR'] = dataframe['AB']/dataframe['TB']
    dataframe = dataframe.drop(['AB','TB','temp'], axis = 1)
    return dataframe

def NAR(dataframe:DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['NAR'] = dataframe['temp'].apply(lambda m: Descriptors.NumAromaticRings(m))
    dataframe = dataframe.drop('temp', axis = 1)
    return dataframe

def FSP3(dataframe: DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    C=Chem.MolFromSmarts('[C;^3]')
    dataframe['Csp3'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(C)))
    TC=Chem.MolFromSmarts('[C,c]')
    dataframe['TC'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(TC)))
    dataframe['FSP3'] = dataframe['Csp3']/dataframe['TC']
    dataframe = dataframe.drop(['Csp3','TC','temp'], axis = 1)
    return dataframe

def PFI(dataframe: DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['NAR'] = dataframe['temp'].apply(lambda m: Descriptors.NumAromaticRings(m))
    dataframe['logP'] = dataframe['temp'].apply(lambda m: Descriptors.MolLogP(m))
    dataframe['PFI'] = dataframe['NAR'] + dataframe['logP']
    dataframe = dataframe.drop(['temp', 'NAR', 'logP'], axis = 1)
    return dataframe

def ArMsp3(dataframe: DataFrame, SmilesCol:str):
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['AA'] = dataframe['temp'].apply(lambda m: len(m.GetAromaticAtoms()))
    C=Chem.MolFromSmarts('[C;^3]')
    dataframe['Csp3'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(C)))
    dataframe['Ar-Sp3'] = dataframe['AA'] - dataframe['Csp3']
    dataframe = dataframe.drop(['AA', 'Csp3', 'temp'], axis = 1)
    return dataframe

def HBondDonors(dataframe:DataFrame, SmilesCol:str):
    nh2=Chem.MolFromSmarts('[n,N;H2]')
    nh1=Chem.MolFromSmarts('[n,N;H1]')
    oh=Chem.MolFromSmarts('[OH]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['N-H'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(nh1)) + 2*len(m.GetSubstructMatches(nh2)))
    dataframe['O-H'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(oh)))
    dataframe['H-Bond Donors'] = dataframe['N-H'] + dataframe['O-H']
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe


def CCCO(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[C,c]=,:[C,c]-,:[C,c]=O')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['C=C-C=O'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe

def CCCC(dataframe:DataFrame, SmilesCol:str):
    s = Chem.MolFromSmarts('[C,c]=,:[C,c]-,:[C,c]=,:[C,c]')
    s2 = Chem.MolFromSmarts('cccc(=O)')
    s3 = Chem.MolFromSmarts('ccc(=O)c')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['C=C-C=C'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)) - len(m.GetSubstructMatches(s2)) - len(m.GetSubstructMatches(s3)) )
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe

def CNCO(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[C,c]=,:[n,N;X2]-,:[C,c]=O')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['C=N-C=O'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe

def CNNC(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[C,c]=,:[n,N;X2]-,:[n,N;X2]=,:[C,c]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['C=N-N=C'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 


def CCCN(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[C,c]=,:[C,c]-,:[C,c]=,:[N,n;X2]')
    s2=Chem.MolFromSmarts('ccc(=[A])[n;X2]')
    s3=Chem.MolFromSmarts('c(=[A])cc[n;X2]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['C=C-C=N'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s))- len(m.GetSubstructMatches(s2)) -len(m.GetSubstructMatches(s3)))
    dataframe=dataframe.drop(['temp'], axis=1)
    return dataframe

def NCCO(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[N,n;X2]=,:[C,c]-,:[C,c]=O')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['N=C-C=O'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 

def NNCC(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[N,n;X2]=,:[N,n,X2]-,:[C,c]=,:[C,c]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['N=N-C=C'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 

def NCCN(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[N,n;X2]=,:[C,c]-,:[C,c]=,:[N,n;X2]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['N=C-C=N'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 

def NNCO(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[N,n;X2]=,:[N,n;X2]-,:[C,c]=O')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['N=N-C=O'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 

def CNCC(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[C,c]=,:[N,n;X2]-,:[C,c]=,:[C,c]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['C=N-C=C'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 

def CNCN(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[C,c]=,:[N,n;X2]-,:[C,c]=[N,n;X2]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['C=N-C=N'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 

def NNCN(dataframe:DataFrame, SmilesCol:str):
    s=Chem.MolFromSmarts('[N,n;X2]=,:[N,n;X2]-,:[C,c]=,:[N,n;X2]')
    PandasTools.AddMoleculeColumnToFrame(dataframe, SmilesCol, 'temp')
    dataframe['N=N-C=N'] = dataframe['temp'].apply(lambda m: len(m.GetSubstructMatches(s)))
    dataframe=dataframe.drop('temp', axis=1)
    return dataframe 
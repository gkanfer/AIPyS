import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tqdm import tqdm


class mapSgRNA():
    def __init__(self,df1,df2):
        '''
        :param df1: Q1 sample (original)
        :param df2: Q2 sample (active sort)
        '''
        self.df1 = df1
        self.df2 = df2
        # adding 2 to all the data for preforming log transformation
        self.df1[self.df1.columns[2]] = self.df1[self.df1.columns[2]] + 2
        self.df2[self.df1.columns[2]] = self.df2[self.df2.columns[2]] + 2
    def mapping(self, uniqueSgRNA):
        '''
        :param uniqueSgRNA: list of strings, unique sgRNA
        :return: mapping reads according to unique sgRNA
        '''
        sgRNAdic = {'Gene':[],'sgRNA':[],"reads_ctrl":[],"reads_activate":[]}
        for sgRNA in tqdm(uniqueSgRNA):
            sgRNAdic['sgRNA'].append(sgRNA)
            # check for non targeting:
            if re.match("non-targeting.*", sgRNA) is None:
                sgRNAdic['Gene'].append(re.sub('_._.*','', sgRNA))
            else:
                sgRNAdic['Gene'].append("non")
            # reads ctrl
            if self.df1.loc[self.df1['sgID'] == sgRNA, self.df1.columns[2]].any():
                sgRNAdic['reads_ctrl'].append(self.df1.loc[self.df1['sgID'] == sgRNA, self.df1.columns[2]].values[0])
            else:
                sgRNAdic['reads_ctrl'].append(2)
            # reads active
            sgRNAdic['reads_activate'].append(self.df2.loc[self.df2['sgID'] == sgRNA, self.df2.columns[2]].values[0])
        return sgRNAdic

    @staticmethod
    def dataFrameFinal(sgRNAdic):
        sgRNAdic['log2FoldChange'] = np.log(sgRNAdic['reads_activate']) - np.log(sgRNAdic['reads_ctrl'])
        sgRNAdic['scaleLog2FoldChange'] = preprocessing.scale(sgRNAdic['log2FoldChange'])
        df = pd.DataFrame(sgRNAdic)
        return df


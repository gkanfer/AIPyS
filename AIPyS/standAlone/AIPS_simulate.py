'''
    Simulation of read counts using negative binomial distribution and preforming AIPS sorting and selecting cells.
    The effective sgRNA are selectd from a pool of sgRNA targeting genes. The number of True positive, FP rate and sample size are predetermined.
'''


import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib as plt
import random
import tqdm as tqdm

class Simulate():
    def __init__(self,df,lookupString = 'PEX',tpRatio  =10,n = 3, p=0.1):
        '''
        :param lookupString: str, substring of target Gene
        :param tpRatio, effective sgRNA number
        :param n, float, number of failures until the experiment is stopped
        :param n, float[0,1],success probability in each experiment
        '''
        self.df = df
        self.lookupString = lookupString
        self.tpRatio = tpRatio
        self.effectiveGuide = self.truePositiveTuple()
        self.n = n
        self.p = p
        self.dfSim = self.observePerRaw()

    def truePositiveTuple(self):
        '''
        :return tuple, sublist of effective sgRNA
        '''
        indexTarget = self.df.loc[self.df.gene.str.contains(self.lookupString)].index.to_list()
        self.df['activeSg'] = False
        indexPexActiveArray = self.df.loc[random.sample(indexTarget, self.tpRatio)].index.tolist()
        self.df.loc[indexPexActiveArray, 'activeSg'] = True
        # list of sgRNA which are true
        TruePositiveSGs = tuple(self.df.loc[indexPexActiveArray, 'sgID'].to_list())
        return TruePositiveSGs

    def observePerRaw(self):
        ''':param
        '''
        self.df['count_sim'] = np.random.negative_binomial(self.n, self.p, len(self.df))
        initSgRNA = self.df.sgID.values.tolist()
        initCount = self.df.count_sim.values.tolist()
        sgRNA = []
        Activity = []
        for sg,count in zip(initSgRNA,initCount):
            sgRNA += [sg]*count
            #
            if sg in self.effectiveGuide:
                Activity += [True]*count
            else:
                Activity += [False] * count
        Qoriginal = pd.DataFrame({'sgID': sgRNA, 'Active': Activity})
        table = Qoriginal.sample(frac=1, replace=False, random_state=1564743454, axis=0, ignore_index=True)
        return table

    def simulation(self,FalseLimits, ObservationNum):
        '''
        :param FalseLimits, tuple, precantage list of False Positive
        :param ObservationNum, tuple, mean and standard deviation
        '''
        Original = self.dfSim
        dfQ1 = {}
        dfQ2 = {}
        fpRate = [arr for arr in np.arange(FalseLimits[0], FalseLimits[1], FalseLimits[0])]
        progress = tqdm.tqdm()
        while True:
            progress.update()
            FOV = int(np.random.normal(ObservationNum[0],ObservationNum[1]))
            if FOV > len(self.dfSim):
                break
            dfTemp = self.dfSim.iloc[:FOV,:]
            # shorten the table by fov
            self.dfSim = self.dfSim.iloc[FOV+1:,:]
            idxTruePostive = dfTemp.index[dfTemp['Active']].tolist()
            if len(idxTruePostive) > 0:
                TruePositiveSGs = dfTemp.loc[idxTruePostive, 'sgID'].to_list()
                dfTemp = dfTemp.drop(idxTruePostive)
                for sg in TruePositiveSGs:
                    if sg in dfQ2.keys():
                        dfQ2[sg] += 1
                    else:
                        dfQ2[sg] = 1
            selFP = int(len(dfTemp.index.to_list()) * random.sample(fpRate,1)[0])
            if selFP > 0:
                TruePositiveSGs =  dfTemp['sgID'].sample(n = selFP).to_list()
                for sg in TruePositiveSGs:
                    dfTemp = dfTemp[dfTemp.sgID != sg]
                    if sg in dfQ2.keys():
                        dfQ2[sg] += 1
                    else:
                        dfQ2[sg] = 1
            sgRNAexclude =  dfTemp['sgID'].to_list()
            for sg in sgRNAexclude:
                if sg in dfQ1.keys():
                    dfQ1[sg] += 1
                else:
                    dfQ1[sg] = 1

        return Original,dfQ1,dfQ2

















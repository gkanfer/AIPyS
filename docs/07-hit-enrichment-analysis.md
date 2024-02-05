Our platform allows for screening rates of up to one million cells per run, with a minimum rate of 100,000 cells. To optimize this process, an enrichment strategy has been developed.

In a CRISPR perturbation screen, the Geometric Distribution is the most commonly used sgRNA sequencing distribution, with an emphasis on evenly distributes sgRNAs across a gene or set of genes. However, the Negative Binomial Distribution is also used and provides more accurate measurement of perturbation by emphasizing the number of times a gene target is hit. 

The ```Simulate``` module simulates AIPS by definingthe  the sgRNA pool for a given gene using the ```lookupString``` parameter. It uses the published crispri sgRNA data (Horlbeck et al) and a Negative Binomial distribution with parameters n and p to determine the number of effective (successful) sgRNA in the pool , as shown in the Python library [pymc](https://www.pymc.io/welcome.html). The argument ```tpRatio``` indicates the number of effective target sgRNA in the pool. It is known that approximately 0.2 to 0.4 of the sgRNA targeting a particular gene are effective, according to the paper by Daley et al. 

- Max A Horlbeck, Luke A Gilbert, Jacqueline E Villalta, Britt Adamson, Ryan A Pak, Yuwen Chen, Alexander P Fields, Chong Yon Park, Jacob E Corn, Martin Kampmann, Jonathan S Weissman (2016) Compact and highly active next-generation libraries for CRISPR-mediated gene repression and activation eLife 5:e19760 https://doi.org/10.7554/eLife.19760

- Daley, T., Lin, Z., Lin, X. et al. CRISPhieRmix: a hierarchical mixture model for CRISPR pooled screens. Genome Biol 19, 159 (2018). https://doi.org/10.1186/s13059-018-1538-6



```python
from AIPyS import AIPS_simulate as sim
# filtered list of sgRNA data from the reference library for the sub library h4
dfH4 = pd.read_csv('dfH4.csv')
```

The command ```simulation``` defines the false positives limits and the number of observations per acquisition, which is drawn from a normal distribution.

```python
orig, Q1, Q2 = sim.Simulate(df = dfH4 ,lookupString =  'PEX' ,tpRatio = 20, n= 10 , p = 0.1).simulation(
                            FalseLimits  = (0.1,0.5), ObservationNum = (70,20))
```
This function returns three data frames: ```orig``` contains the original read counts, ```Q1``` contains the read counts of the sgRNAs not selected during acquisition, and ```Q2``` contains the read counts of the selected hits the "activated" sample.


```python
self.df['activeSg'] = False
indexPexActiveArray = self.df.loc[random.sample(indexTarget, self.tpRatio)].index.tolist()
self.df.loc[indexPexActiveArray, 'activeSg'] = True
# list of sgRNA which are true
TruePositiveSGs = tuple(self.df.loc[indexPexActiveArray, 'sgID'].to_list())
```

Enrichment analysis was performed by comparing the activated simulated data (Q2) with the non-activated simulated data (Q1). Mapping of the read counts was done according to the unique Gene names, and the perturbation effects were calculated by dividing the log count data of Q2 by that of Q1.

```python
from AIPyS import mapSgRNA 
# filtered list of sgRNA data from the reference library for the sub library h4
pathIn = r'F:\HAB_2\PrinzScreen\AIPS_simulation\AIPS_simulation\data'
Q2 = pd.read_csv(os.path.join(pathIn,'h4_1_45tpGuide_Q2.csv'))
Q1 = pd.read_csv(os.path.join(pathIn,'h4_1_45tpGuide_Q1.csv'))
# unique

uniqueSgRNA = np.unique(Q2.sgID.values)

# The mapping of unique sgRNA to the SIM was performed using the module mapsgrna

dictDF = mapSgRNA(df1 = Q1, df2 = Q2)

# Make dictionary mapping reads to unique sgRNA

df_dict = dictDF.mapping(uniqueSgRNA = uniqueSgRNA)

# data frame contain logFC and z-score logFC

df = dictDF.dataFrameFinal(df_dict)
```

<center><b><u>Simulation</u></b></center>

![png](output_7_0_0.png)
    

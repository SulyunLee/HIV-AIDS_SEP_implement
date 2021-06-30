# Predictin of HIV/AIDS infections in Syringe Exchange Program Implementation
## Summary
There is an ongoing comfliction on the implementation of syringe dispensing and disposal program (also known as syringe exchange program, or SEP) for people who inject drugs. Some believe that the cost of implementing the program is expensive compared to the possible benefit. However, we argue that implementing the SEPs would significantly reduce the HIV/AIDS infections by providing sanitary needles and accessible disposal programs. We aim to analyze the national impact of SEPs in the United States when they are legalized in state-level. We predicted the numbers of HIV/AIDS infections and deaths when SEPs are legalized or when they are made illegal in every state. We also experimented on a case study in Iowa, where only two SEPs exist in the state but not legalized.

## Data
We collected data for HIV/AIDS diagnoses and deaths due to injectable drug use from CDC AtlasPlus (https://www.cdc.gov/nchhstp/atlas). We also collected the data for the number of each state's available and legality of SEPs from The Foundation for AIDS Research (amFAR) (https://www.amfar.org).

## Model description
### Variables
* Outcome Variable: HIV diagnoses per 10,0000, HIV deaths per 10,0000, AIDS diagnoses per 10,0000, AIDS deaths per 10,0000
* Predictors: Number of SEPs or SEP legality in each state
* Confounders: rate of population without a high schooldiploma, rate of uninsured individuals, rate of each race (American Indian/Alaskan Native, Asian, Black/African American, Native Hawaiian/Other pacific islander, and White), comorbidity of syphilis cases, Ryan White (federal funding for HIV/AIDS treatment) client rate.

### ML models
Linear regression and random forest

### Simulation
We experimented on different scenarios of SEP implementation and used the simulated SEP implementation variables to predict the HIV/AIDS diagnoses and deaths.
* Number of SEPs in each state
1. *Scenario 1*: Number of SEPs is reduced to zero (No SEP in a state)
2. *Scenario 2*: Number of SEPs is reduced to half
3. *Scenario 3*: Number of SEPs is increased by double

* SEP legality in each state
1. *Scenario 1*: SEP legality is all changed to 'not legal'
2. *Scenario 2*: SEP legality is all changed to 'legal'

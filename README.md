# Prediction of HIV/AIDS infections in Syringe Exchange Program Implementation
## Summary
There is an ongoing conflict on the implementation of a syringe dispensing and disposal program (also known as a syringe exchange program, or SEP) for people who inject drugs. Some believe that the cost of implementing the program is expensive compared to the possible benefit. However, we argue that implementing the SEPs would significantly reduce HIV/AIDS infections by providing sanitary needles and accessible disposal programs. We aim to analyze the national impact of SEPs in the United States when they are legalized at the state level. We predicted the numbers of HIV/AIDS infections and deaths when SEPs are legalized or when they are made illegal in every state. We also experimented on a case study in Iowa, where only two SEPs exist in the state but are not legalized.

## Data
We collected data for HIV/AIDS diagnoses and deaths due to injectable drug use from CDC AtlasPlus (https://www.cdc.gov/nchhstp/atlas). We also collected the data for the number of each state's available SEPs and legality of SEPs from The Foundation for AIDS Research (amFAR) (https://www.amfar.org). We used the state-level data between 2011 and 2017 to train the model for the prediction.

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

### Results
* Simulation results of HIV/AIDS diagnosis and death prediction when the number of SEPs is changed

|HIV diagnosis|HIV death|
|-------------|---------|
|![alt](https://user-images.githubusercontent.com/12605926/124176858-19929e00-da75-11eb-80b0-2dbb1e993a64.png)|![alt](https://user-images.githubusercontent.com/12605926/124176881-22836f80-da75-11eb-9275-62e9daa23499.png)|

|AIDS diagnosis|AIDS death|
|--------------|----------|
|![alt](https://user-images.githubusercontent.com/12605926/124178396-20221500-da77-11eb-9f4b-3c752bf78f71.png)|![alt](https://user-images.githubusercontent.com/12605926/124176992-45158880-da75-11eb-855c-974d16666520.png)|

* Simulation results of HIV/AIDS diagnosis and death prediction when the SEP legality is changed

|HIV diagnosis|HIV death|
|-------------|---------|
|![alt](https://user-images.githubusercontent.com/12605926/124178717-68d9ce00-da77-11eb-81f0-98b6445dc718.png)|![alt](https://user-images.githubusercontent.com/12605926/124178735-70997280-da77-11eb-8044-8a877203e790.png)|

|AIDS diagnosis|AIDS death|
|--------------|----------|
|![alt](https://user-images.githubusercontent.com/12605926/124178649-52cc0d80-da77-11eb-9a8c-ffc3f71b4a34.png)|![alt](https://user-images.githubusercontent.com/12605926/124178676-59f31b80-da77-11eb-9826-5fa34d2cd049.png)|



### Implications
The simulation of changing the number of SEPs clearly shows that when the number of SEPs is increased by double for all states, the number of HIV/AIDS diagnoses and deaths decreases in each year. In average, the simulation showed that the national number of HIV/AIDS diagnoses can be reduced by 6% in each year and the national number of HIV/AIDS deaths can be reduced by 7% in each year. However, the legalization does not seem to help reduce the HIV/AIDS diagnosis and death. In summary, lawmakers in states would significantly reduce the infections of HIV/AIDS through continuously increrasing the locations of SEPs rather than spending more time and money in passing the legislation.

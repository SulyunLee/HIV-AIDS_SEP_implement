import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def plot_num_ssp(model, outcome):
    filename = '{}{}_{}_numSSPs_simulation_result.csv'.format(results_dir, model, outcome)
    result = pd.read_csv(filename)
    ax0 = sns.lineplot(x='Year', y='Actual', data=result, marker='o')
    sns.lineplot(x='Year', y='Simulated zero', data=result, marker='o')
    sns.lineplot(x='Year', y='Simulated half', data=result, marker='o')
    sns.lineplot(x='Year', y='Simulated double', data=result, marker='o')
    ax0.lines[1].set_linestyle("--")
    ax0.lines[2].set_linestyle("--")
    ax0.lines[3].set_linestyle("--")
    plt.legend(labels=['Actual data', 'zero SSPs', 'SSPs reduced to half', 'SSPs increased two times'])
    plt.xlim(result.Year.min(), result.Year.max())
    plt.ylabel(outcome)
    plt.title("Simulation of {} with the number of SSPs changes".format(outcome))
    plt.grid(color = 'lightgray')
    plt.savefig("{}{}_{}_numSSPs_simulation.png".format(plots_dir, model, outcome))
    plt.close()

def plot_ssp_legality(model, outcome):
    filename = '{}{}_{}_SSPLegality_simulation_result.csv'.format(results_dir, model, outcome)
    result = pd.read_csv(filename)
    ax0 = sns.lineplot(x='Year', y='Actual', data=result, marker='o')
    sns.lineplot(x='Year', y='All SSP illegal', data=result, marker='o')
    sns.lineplot(x='Year', y='All SSP legal', data=result, marker='o')
    ax0.lines[1].set_linestyle("--")
    ax0.lines[2].set_linestyle("--")
    plt.legend(labels=['Actual data', 'All SSPs are illegal', 'All SSPs are legal'])
    plt.xlim(result.Year.min(), result.Year.max())
    plt.ylabel(outcome)
    plt.title("Simulation of {} with the SSPs legality changes".format(outcome))
    plt.grid(color = 'lightgray')
    plt.savefig("{}{}_{}_SSPLegality_simulation.png".format(plots_dir, model, outcome))
    plt.close()

if __name__ == "__main__":
    data_dir = "cleaned_dataset/"
    results_dir = "results/"
    plots_dir = "plots/"

    sns.set_palette("deep")

    df = pd.read_csv(data_dir + "cleaned_final_dataset.csv")

    # total number in states
    hiv_aids_summary = df.groupby('Year', as_index=False).agg({'HIV diagnoses':'sum', 'HIV deaths':'sum',
                                                                'AIDS diagnoses':'sum', 'AIDS deaths':'sum'})

    # total number of SSPs and SSP legality in states
    ssp_summary = df.groupby('Year', as_index=False).agg({'NUMBER OF SSPs':'sum', 'SSP LEGALITY BINARY':'sum'})



    # Nebraska vs. Kentucky
    neb = df[df['Geography'] == 'Nebraska']
    ken = df[df['Geography'] == 'Kentucky']
    sns.lineplot(x='Year', y='NUMBER OF SSPs', data=ken, marker='o')
    plt.xlim(2011,2017)
    plt.ylabel("Number of SSPs")
    plt.title("The number of SSPs in states with highest increase")
    plt.grid(color='lightgray')
    plt.savefig(plots_dir + "num_ssp_Kentucky.png")
    plt.close()

    # line graph of the number of SSPs (all states)
    sns.lineplot(x='Year', y='NUMBER OF SSPs', data=ssp_summary, marker='o') 
    plt.xlim(2011, 2017)
    plt.ylabel("Number of SSPs")
    plt.title("The total number of SSPs in the United States by years")
    plt.grid(color='lightgray')
    plt.savefig(plots_dir + "num_ssp_allstates_by_year.png")
    plt.close()

    # line graph of the SSP legality (all states)
    sns.lineplot(x='Year', y='SSP LEGALITY BINARY', data=ssp_summary, marker='o') 
    plt.xlim(2011, 2017)
    plt.ylabel("The number of states with legal SSPs")
    plt.title("The total number of states with legal SSPs in the United States by years")
    plt.grid(color='lightgray')
    plt.savefig(plots_dir + "ssp_legality_allstates_by_year.png")
    plt.close()

    # plot simulation results
    for model in ['rf', 'linreg']:
        for outcome in ['HIV diagnoses per 100000', 'HIV deaths per 100000', 'AIDS diagnoses per 100000', 'AIDS deaths per 100000']:
            plot_num_ssp(model, outcome)
            plot_ssp_legality(model, outcome)


    # state simulation
    hiv_diagn_state_df = pd.read_csv("linreg_state_simulation_HIV diagnoses per 100000.csv")
    hiv_death_state_df = pd.read_csv("linreg_state_simulation_HIV deaths per 100000.csv")
    aids_diagn_state_df = pd.read_csv("linreg_state_simulation_AIDS diagnoses per 100000.csv")
    aids_death_state_df = pd.read_csv("linreg_state_simulation_AIDS deaths per 100000.csv")

    sns.lineplot(x="Year", y="HIV diagnoses per 100000", data=hiv_diagn_state_df, marker='o')
    # sns.lineplot(x="Year", y="Arkansas simulation", data=hiv_diagn_state_df, marker='o')
    sns.lineplot(x="Year", y="HIV deaths per 100000", data=hiv_death_state_df, marker='o')
    sns.lineplot(x="Year", y="AIDS diagnoses per 100000", data=aids_diagn_state_df, marker='o')
    sns.lineplot(x="Year", y="AIDS deaths per 100000", data=aids_death_state_df, marker='o')
    plt.xlim(2020,2026)
    plt.ylabel("HIV/AIDS diagnoses and deaths per 100,000")
    plt.title("The prediction of HIV/AIDS in Iowa with number of SSPs in Connecticut")
    plt.grid(color='lightgray')
    plt.legend(labels=["HIV diagnoses per 100000", "HIV deaths per 100000",
                        "AIDS diagnoses per 100000", "AIDS deaths per 100000"])
    plt.savefig(plots_dir+"num_SSPs_Iowa_simulation.png")
    
    plt.close()
    

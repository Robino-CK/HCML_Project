import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(distributions, header):
    
    # plot beta distribution
    plt.hist(distributions, bins=100, color="#C00A35")
    plt.title(header)
    plt.savefig(f"res/plots/hist_{header}.png")
    plt.show()
    
def plot_bar(shap_values, ebm_values, pfi_values, columns, header="test"):
    #fig, ax = plt.subplots()
    df = pd.DataFrame({"feature" : columns, "EBM": ebm_values,"NN (SHARP)": shap_values , "NN (PFI)" : pfi_values} )
    df.sort_values(by=["NN (SHARP)"], inplace = True)
    
    df.plot.barh(x = 'feature', y = ['EBM', 'NN (SHARP)', "NN (PFI)"], color={"EBM": "#FFCD00", "NN (SHARP)": "#AA1555", "NN (PFI)": "#5287C6"})
    plt.xlabel("Feature Importance")
    plt.yticks(rotation=45)
    plt.ylabel("Features")
    plt.legend(loc='right')
    #ax.legend("EBM": "Feature Importance EBM","NN": "Feature Importance NN"}
    plt.savefig(f"res/plots/barh_{header}.png")
    plt.show()
    
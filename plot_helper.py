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
    df.sort_values(by=["feature"], ascending = False, inplace = True)
    
    df.plot.barh(x = 'feature', y = ['Normalgaus'], color={"Normalgaus": "#FFCD00", "Normalgaus": "#AA1555", "": "#5287C6"})
    plt.xlabel("Feature Importance")
    plt.yticks(rotation=45)
    plt.ylabel("Features")
    plt.legend(loc='right')
    #ax.legend("EBM": "Feature Importance EBM","NN": "Feature Importance NN"}
    plt.savefig(f"res/plots/barh_{header}.png")
    plt.show()
    
    
def plot_bar_per_scale(metric_values,columns,experiments=["Baseline", "Normaldistribution", "Weibulldistribution"], experiments_names = None,  header="test"):
    #fig, ax = plt.subplots()
    if not experiments_names:
        experiments_names = experiments
    d = {}
    color_list = {"#FFCD00" ,"#AA1555", "#5287C6", "#000000"}
    d_colors = {}
    
    for exp_n,exp in zip(experiments_names,experiments):
        
        
        d[exp_n] = metric_values[f"{exp}"]
        d_colors[exp_n] = color_list.pop()
        if color_list == set():
            color_list.add("#000000")
        
    
    d["feature"] = columns
    df = pd.DataFrame(d )
    df.sort_values(by=["feature"], ascending = False, inplace = True)
    
    df.plot.barh(x = 'feature', y = experiments_names , color=d_colors, title=header)
    plt.xlabel("Feature Importance")
    plt.yticks(rotation=45)
    plt.ylabel("Features")
    plt.legend(loc='right')
    #ax.legend("EBM": "Feature Importance EBM","NN": "Feature Importance NN"}
    plt.savefig(f"res/plots/barh_{header}.png")
    plt.show()
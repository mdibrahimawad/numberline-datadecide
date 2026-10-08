## Table 1 Results: montonocity sublinearity and explain variance (PCA) or R^2 (PLS)
save_path = 'ICLR_results'
import sys
if len(sys.argv) > 1:
    model_key = sys.argv[1]
else:
    raise ValueError("Enter  model Name")

from utils.data_utils import search_models
model_name = model_key if "/" in model_key else search_models(model_key)
if model_name is None:
    raise ValueError(f"Could not resolve model key: {model_key}")
# Example: print(search_models("llama"))

from dataclasses import dataclass
@dataclass
class ArgumentParser:
    transform: str = "PCA"              # dimensionality reduction method: "PCA" or "PLS"
    Tdim: int = 1                       # number of target dimensions after transform
    k: int = 30                         # group/examples
    num_examples: int = 3               # number of demonstrations
    context: str = 'paper'              # paper: context numbers sampled from the same Gi as x
    data: str = 'numerics'              # dataset type to use numerics or symbols
    groups = [1,2,3,4]   # groups to test; here each group is 10**i
    upper_bound: int = 10**groups[-1]   # upper bound for generated numbers (max context size)
    save: bool = True                   # whether to save results
    plot: bool = True                   # whether to plot results
    model_name: str = model_name  # model identifier (e.g., huggingface repo name)
    device: str = "0"                   # GPU device number (string)
    runs: int = 3                       # number of runs
    interval_mode: str = "paper"        # paper Eq. 4 intervals
    beta_method: str = "geometric"      # paper Eq. 3 geometric regression
    filter_correct: bool = True         # keep x only when model generates x after final =
    filter_data: str = "numerics"       # avoid over-filtering random-letter controls by default
    filter_max_new_tokens: int = 8
    letter_mode: str = "paper_token_length"
args = ArgumentParser()

from main_lab import analyzer
res = analyzer(args)

def print_layer_metrics(layer, metrics):
    print(f"{'Layer':<10} {'EV (mean±std)':<22} {'rho (mean±std)':<22} {'beta (mean±std)':<22}")
    result_line = (f"{layer:<10} "
                   f"{metrics['EV']:.2f} ± {metrics['EV_std']:.3f}{' ' * 13}"
                   f"{metrics['rho']:.2f} ± {metrics['rho_std']:.3f}{' ' * 10}"
                   f"{metrics['beta']:.2f} ± {metrics['beta_std']:.3f}")
    print(result_line)
    return result_line
import os
path_PCA = os.path.join(save_path, "table_1_PCA.txt")
path_PLS = os.path.join(save_path, "table_1_PLS.txt")

#1 run for numerics
res.groups = [1,2,3,4]
res.data ='numerics'
results = res.run_multiple()
#Table 1 line results[0]
print(args.model_name)
print(res.data)

results_sort = sorted(results[0].items(), key=lambda item: item[1]['EV'],reverse=True)
layer, metrics = results_sort[0]
# prepare the results in table format
result_line = print_layer_metrics(layer, metrics)



with open(path_PCA, "a") as f:
    f.write(f"{args.model_name}\n")
    f.write(f"{res.data}\n")
    f.write(f"{'Layer':<10} {'EV (mean±std)':<22} {'rho (mean±std)':<22} {'beta (mean±std)':<22}\n")
    f.write(result_line + "\n")

#Table 1_PLS line results[1]
print(args.model_name)
print(args.data)
results_sort = sorted(results[1].items(), key=lambda item: item[1]['EV'],reverse=True)
layer, metrics = results_sort[0]
result_line = print_layer_metrics(layer, metrics)



# # Append result to table_1_PLS.txt
with open(path_PLS, "a") as f:
    f.write(f"{args.model_name}\n")
    f.write(f"{res.data}\n")
    f.write(f"{'Layer':<10} {'EV (mean±std)':<22} {'rho (mean±std)':<22} {'beta (mean±std)':<22}\n")
    f.write(result_line + "\n")


#2 run for controlled letters instaed of numerics:
res.data ='letters'
results = res.run_multiple()
#Table 1 line
print(args.model_name)
print(res.data)

results_sort = sorted(results[0].items(), key=lambda item: item[1]['EV'],reverse=True)
layer, metrics = results_sort[0]
result_line = print_layer_metrics(layer, metrics)

# Append result to table_1.txt
with open(path_PCA, "a") as f:
    f.write(f"{args.model_name}\n")
    f.write(f"{res.data}\n")
    f.write(f"{'Layer':<10} {'EV (mean±std)':<22} {'rho (mean±std)':<22} {'beta (mean±std)':<22}\n")
    f.write(result_line + "\n")

#Table 2 line for PLS
print(args.model_name)
print(args.data)
results_sort = sorted(results[1].items(), key=lambda item: item[1]['EV'],reverse=True)
layer, metrics = results_sort[0]
result_line = print_layer_metrics(layer, metrics)

# # Append result to table_2.txt
with open(path_PLS, "a") as f:
    f.write(f"{args.model_name}\n")
    f.write(f"{res.data}\n")
    f.write(f"{'Layer':<10} {'EV (mean±std)':<22} {'rho (mean±std)':<22} {'beta (mean±std)':<22}\n")
    f.write(result_line + "\n")

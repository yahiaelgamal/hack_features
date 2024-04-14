# %%
# %pip install transformer_lens # sae_lens
# %pip install matplotlib numpy scipy plotly pytest

# %% [markdown]
# # Setup

# %%
from fancy_einsum import einsum
import pandas as pd

import torch
import plotly.express as px
import matplotlib.pyplot as plt
from sae_lens.training.session_loader import LMSparseAutoencoderSessionloader
from huggingface_hub import hf_hub_download
# from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
# TOKENIZERS_PARALLELISM
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# %% [markdown]
# ### Get data

# %%
dataset = load_dataset("apollo-research/Skylion007-openwebtext-tokenizer-gpt2", split='train', streaming=True)
shuffled_dataset = dataset.shuffle(seed=42, buffer_size=100)
dataloader = DataLoader(shuffled_dataset, batch_size=10)

# %%
# Start by downloading them from huggingface
device = "cuda"
REPO_ID = "jbloom/GPT2-Small-SAEs"

def get_sae(layer):
    assert 0<=layer<12, "Layer must be between 0 and 11"
    FILENAME = f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"

    # this is great because if you've already downloaded the SAE it won't download it twice!
    path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

    # We can then load the SAE, dataset and model using the session loader
    model, sparse_autoencoders, activation_store = (
        LMSparseAutoencoderSessionloader.load_session_from_pretrained(path=path)
    )
    assert len(list(sparse_autoencoders)) == 1, "There should only be one SAE in this file"
    return model, list(sparse_autoencoders)[0], activation_store

# %%
model, _, activation_store = get_sae(layer=8)
saes = []
for layer in range(12):
    model, sae, activation_store = get_sae(layer=layer)
    saes.append(sae)

n_features = sae.cfg.d_sae

# %% [markdown]
# # Collect SAE feature activations

# %%
def get_feature_activation_table(sae, dataloader=dataloader, batches=10):
    layer = sae.cfg.hook_point_layer
    big_feature_activation_table = []
    sae.eval()  # prevents error if we're expecting a dead neuron mask for who grads
    for i, prompts in tqdm(enumerate(iter(dataloader)), desc=f"Getting activations for layer {layer}", total=batches):
        with torch.no_grad():
            prompts = torch.stack(prompts["input_ids"]).to(device)
            _, cache = model.run_with_cache(prompts, prepend_bos=True)
            _, feature_acts, _, _, _, _ = sae(
                cache[sae.cfg.hook_point]
            )
            del cache
            big_feature_activation_table.append(feature_acts[:, 1:, :].reshape(-1, n_features).cpu())
        if i==batches:
            break
    big_feature_activation_table = torch.cat(big_feature_activation_table, dim=0)
    return big_feature_activation_table

torch.cuda.empty_cache()
feature_activation_tables = []
for sae in saes:
    feature_activation_tables.append(get_feature_activation_table(sae))

n_data = feature_activation_tables[0].shape[0]


# %%

for activation_threshold in [0.5]:#, 0.01, 0.05, 0.1, 0.9]:
    print(f"Activation threshold: {activation_threshold}")

    torch.cuda.empty_cache()
    binary_feature_activation_tables = []
    for layer in tqdm(range(12), desc="Creating binary feature activation tables"):
        binary_feature_activation_tables.append((feature_activation_tables[layer] > feature_activation_tables[layer].max(dim=0).values*activation_threshold).to(torch.float32))

    torch.cuda.empty_cache()
    B_given_As = []
    A_given_Bs = []
    jaccards = []
    pwmis = []
    for layer in tqdm(range(11), desc="Computing cooccurence fractions and pointwise mutual information", position=0):
        data_batch_size = 50
        cooccurence = 0
        unionoccurrence = 0
        occurrence_A = binary_feature_activation_tables[layer].mean(dim=0)
        occurrence_B = binary_feature_activation_tables[layer+1].mean(dim=0)
        
        for data_batch in tqdm(range(0, n_data, data_batch_size), desc="Batches", leave=False, position=1):
            data_slice = slice(data_batch, min(data_batch+data_batch_size, n_data))
            # Jaccard is too hard because computing the Union is hard
            a = binary_feature_activation_tables[layer][data_slice].to("cuda")
            b = binary_feature_activation_tables[layer+1][data_slice].to("cuda")
            cooccurence += einsum("data feature1, data feature2 -> feature1 feature2", a, b)
        cooccurence = cooccurence.to("cpu") / n_data

        pwmi = torch.log(cooccurence / (occurrence_A.unsqueeze(1)+1e-9) / (occurrence_B.unsqueeze(0)+1e-9))
        pwmis.append(pwmi)

        B_given_A = cooccurence / occurrence_A.unsqueeze(1)
        A_given_B = cooccurence / occurrence_B.unsqueeze(0)
        B_given_As.append(B_given_A)
        A_given_Bs.append(A_given_B)
        
        


    def save_to_pandas_csv(data, filename, filter_threshold=0.001):
        df = {"layer_A": [], "layer_B": [], "feature_A": [], "feature_B": [], "weight": []}
        for layer in tqdm(range(len(data))):
            for i in range(len(data[layer])):
                for j in np.where(data[layer][i] > filter_threshold)[0]:
                    df["layer_A"].append(layer)
                    df["layer_B"].append(layer+1)
                    df["feature_A"].append(i)
                    df["feature_B"].append(j)
                    df["weight"].append(data[layer][i,j].item())
        df = pd.DataFrame(df)
        # filter_threshold based rounding
        # df['weight'] = df['weight'].apply(lambda x: round(x, int(-np.log10(filter_threshold))))
        df.to_csv(f"/mnt/ssd-rib/stefan/hackathon/{filename}")

    filter_threshold = 0.01
    save_to_pandas_csv(B_given_As, f"B_given_As_{n_features}_features_{n_data}_tokens_{activation_threshold}_activationthreshold_{filter_threshold}_filterthreshold.csv", filter_threshold=filter_threshold)
    save_to_pandas_csv(A_given_Bs, f"A_given_Bs_{n_features}_features_{n_data}_tokens_{activation_threshold}_activationthreshold_{filter_threshold}_filterthreshold.csv", filter_threshold=filter_threshold)

    filter_threshold = 0.1
    save_to_pandas_csv(pwmis, f"mutual_information_{n_features}_features_{n_data}_tokens_{activation_threshold}_activationthreshold_{filter_threshold}_filterthreshold.csv", filter_threshold=filter_threshold)

    filter_threshold = 0.1
    save_to_pandas_csv(jaccards, f"jaccard_{n_features}_features_{n_data}_tokens_{activation_threshold}_activationthreshold_{filter_threshold}_filterthreshold.csv", filter_threshold=filter_threshold)

# save_to_pandas_csv(pwmis, f"mutual_information_{n_features}_features_{n_data}_tokens_{activation_threshold}_activationthreshold.csv")

# %%
# # Load cooccurence_fraction_24576_features_101376_tokens_0.5_activationthreshold_0.001_filterthreshold.csv to pabdas
# data = pd.read_csv("/mnt/ssd-rib/stefan/hackathon/cooccurence_fraction_24576_features_101376_tokens_0.5_activationthreshold_0.001_filterthreshold.csv")
# # Filter edge > 0.1
# data = data[data['weight'] > 0.1]
# # Truncate to 2 decimal places
# data['weight'] = data['weight'].apply(lambda x: round(x, 2))
# # Exclude indices
# data = data.drop(columns=['Unnamed: 0'])
# # Save as csv
# data.to_csv("/mnt/ssd-rib/stefan/hackathon/cooccurence_fraction_24576_features_101376_tokens_0.5_activationthreshold_0.1_filterthreshold.csv")
# %%

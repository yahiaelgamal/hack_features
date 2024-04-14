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
from tqdm import tqdm
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
def get_feature_activation_table(sae, dataloader=dataloader, batches=1):
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
    # for _ in tqdm(range(batches), desc=f"Getting activations for layer {layer}"):
    #     with torch.no_grad():
    #         batch_tokens = activation_store.get_batch_tokens()
    #         _, cache = model.run_with_cache(batch_tokens, prepend_bos=True)
    #         _, feature_acts, _, _, _, _ = sae(
    #             cache[sae.cfg.hook_point]
    #         )
    #         del cache
    #         big_feature_activation_table.append(feature_acts.view(-1, n_features).cpu())

    big_feature_activation_table = torch.cat(big_feature_activation_table, dim=0)
    return big_feature_activation_table

feature_activation_tables = []
for sae in saes:
    feature_activation_tables.append(get_feature_activation_table(sae))

n_data = feature_activation_tables[0].shape[0]


# %%

activation_threshold = 0.5

binary_feature_activation_tables = []
for layer in tqdm(range(12), desc="Creating binary feature activation tables"):
    binary_feature_activation_tables.append((feature_activation_tables[layer] > feature_activation_tables[layer].max(dim=0).values*activation_threshold).to(torch.float32))

cooccurence_fractions = []
pwmis = []
for layer in tqdm(range(11), desc="Computing cooccurence fractions and pointwise mutual information"):
    # Jaccard is too hard because computing the Union is hard
    cooccurence = einsum("data feature1, data feature2 -> feature1 feature2", binary_feature_activation_tables[layer], binary_feature_activation_tables[layer+1])
    cooccurence_fraction = cooccurence / n_data
    cooccurence_fractions.append(cooccurence_fraction)
    pwmi = (cooccurence) / (1e-8 + binary_feature_activation_tables[layer].sum(dim=0).unsqueeze(1) * binary_feature_activation_tables[layer+1].sum(dim=0).unsqueeze(0))
    pwmis.append(pwmi)

# %%
# Pandas df
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
    df.to_csv(f"/mnt/ssd-rib/stefan/hackathon/{filename}")

save_to_pandas_csv(cooccurence_fractions, f"cooccurence_fraction_{n_features}_features_{n_data}_tokens_{activation_threshold}_activationthreshold.csv")

save_to_pandas_csv(pwmis, f"mutual_information_{n_features}_features_{n_data}_tokens_{activation_threshold}_activationthreshold.csv")

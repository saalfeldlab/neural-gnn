# neural-gnn

<p align="justify">
Graph neural networks trained to predict observable dynamics can be used to decompose the temporal activity of complex heterogeneous systems into simple, interpretable representations. Here we apply this framework to simulated neural assemblies with thousands of neurons and demonstrate that it can jointly reveal the connectivity matrix, the neuron types, the signaling functions, and in some cases hidden external stimuli. In contrast to existing machine learning approaches such as recurrent neural networks and transformers, which emphasize predictive accuracy but offer limited interpretability, our method provides both reliable forecasts of neural activity and interpretable decomposition of the mechanisms governing large neural assemblies.
</p>

---

### **[Interactive Documentation](https://saalfeldlab.github.io/neural-gnn/)**

<p align="justify">
<strong>Explore the paper's figures and results</strong> - The documentation site provides an interactive walkthrough of all figures from the paper, with detailed explanations of the methodology, training procedures, and learned representations.
</p>

---

<p align="center">
  <img src="./assets/Fig1.png" alt="NeuralGraph Overview" width="500">
</p>
<p align="center"><em>The temporal activity of a simulated neural network (a) is converted into densely connected graph (b) processed by a message passing GNN (c). Each neuron (node i) receives activity signals from connected neurons (node j), processed by a transfer function and weighted by the connection matrix. The sum of these messages is updated to obtain the predicted activity rate. In addition to the observed activity, the GNN has access to learnable latent vectors associated with each node.</em></p>

### Setup

Create a conda environment based on your system architecture:

- MacOS:

```
conda env create -f envs/environment.mac.yaml
conda activate neural-graph-mac
```

- Linux: we are currently using the cuda 13 wheels, which requires that your
  system nvidia drivers are version >= 565.xx. Update the `--extra-index-url`
  to use the cuda 12.x wheels, e.g.,

```
--- a/envs/environment.linux.yaml
+++ b/envs/environment.linux.yaml
@@ -41,8 +41,8 @@ dependencies:
   - jupytext

   - pip:
-      # Get CUDA 13.0 wheels
-      - --extra-index-url https://download.pytorch.org/whl/cu130
+      # Get CUDA 12.9 wheels
+      - --extra-index-url https://download.pytorch.org/whl/cu129
       - torch==2.9
       - torchvision==0.24
       - torchaudio==2.9
```

and then

```
conda env create -f envs/environment.linux.yaml
conda activate neural-gnn
```

Run the first notebook with:

```
python Notebook_01.py

```

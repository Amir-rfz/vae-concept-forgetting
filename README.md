# VAE Concept Forgetting: An Implementation of Selective Amnesia

This repository contains a PyTorch implementation of the paper **"Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models"** presented at NeurIPS 2023. The code is structured as a step-by-step Google Colab notebook that demonstrates how to make a Variational Autoencoder (VAE) forget a specific concept (in this case, an MNIST digit).

**Original Paper:** [arXiv:2305.10120](https://arxiv.org/abs/2305.10120)

## How It Works

The project is broken down into a series of steps, each corresponding to a cell in the provided Google Colab notebook. The process demonstrates how to first train a generative model and then apply the Selective Amnesia technique to make it "unlearn" a specific class.

### Step 1: Initial CVAE Training

First, a standard Conditional Variational Autoencoder (CVAE) is trained on the complete MNIST dataset, containing all 10 digits (0-9). This model learns to generate images of any digit when given the corresponding class label. The progress of this training process can be seen below:

<p align="center">
  <img src="https://github.com/Amir-rfz/vae-concept-forgetting/blob/main/additional_files/VAE%20Learning%20Process.gif" 
       alt="VAE Learning Process" 
       style="width: 450px; border: 1px solid #ddd; border-radius: 8px;">
</p>

### Step 2: Calculate Fisher Information Matrix (FIM)

Once the VAE is fully trained, we analyze it to determine the importance of each weight in its network. This is done by calculating the Fisher Information Matrix (FIM). The FIM essentially tells us which connections are crucial for remembering the learned information. This information will be used in the next step to protect the model's memory of the digits we want to keep.

### Step 3: Forgetting Training with Selective Amnesia

This is the core of the project. We take the fully trained VAE and the calculated FIM and begin a second phase of training. In this phase, the model is taught to:

1.  **Forget:** Associate the label of the forgotten class (e.g., '0') with random noise.
2.  **Remember:** Continue to generate clear images for all other classes (1-9).
3.  **Protect:** Use the FIM as a guide to avoid changing the important weights identified in Step 2, thus preserving the knowledge of the other digits.

<p align="center">
  <img src="https://github.com/Amir-rfz/vae-concept-forgetting/blob/main/additional_files/Forgetting%20Process.gif" 
       alt="Forgetting Process" 
       style="width: 450px; border: 1px solid #ddd; border-radius: 8px;">
</p>

### Step 4: Generate Samples for Evaluation

After the forgetting process is complete, we generate two sets of images: one from the original VAE and one from the new "amnesiac" VAE. For both, we ask them to generate images of the digit they were supposed to forget. These saved images are used for the final quantitative analysis.

### Step 5: Quantitative Forgetting Evaluation

To prove that the forgetting was successful, we use an independent, pre-trained MNIST classifier to evaluate the generated samples. We measure two key metrics from the paper:

* **Average Probability:** The classifier's confidence that it's seeing the forgotten digit. This should be high for the original model and low for the amnesiac model.
* **Classifier Entropy:** A measure of the classifier's "confusion." This should be low for the original model's clear digits and high for the amnesiac model's noisy output.

Here are the results from running the evaluation for forgetting the digit '1':

| Model                          | Avg. Probability of seeing a '1' | Classifier Entropy |
| ------------------------------ | -------------------------------- | ------------------ |
| **Original Model (Before)** | 0.9704                           | 0.0999             |
| **Amnesiac Model (After)** | 0.2045                           | 2.2175             |

As expected, the average probability of the target class drops significantly after forgetting, while the classifier's entropy (confusion) increases to near its maximum possible value.

## Results

The visual results clearly show the effect of the Selective Amnesia algorithm. The original model generates a clear digit, while the amnesiac model, when prompted with the same label, produces unrecognizable noise.

<p align="center">
  <img src="https://github.com/Amir-rfz/vae-concept-forgetting/blob/main/additional_files/final_amnesia_comparison.png" 
       alt="final comparison" 
       style="width: 750px; border: 1px solid #ddd; border-radius: 8px;">
</p>

The quantitative results from the classifier evaluation further confirm this, showing a significant drop in the probability of the forgotten class and a large increase in classifier entropy.

## How to Run

The code for this project is contained within a single Google Colab notebook.

1.  Open the notebook in Google Colab.
2.  Ensure the runtime is set to use a GPU for faster training.
3.  Run each cell in order, from top to bottom. The notebook is designed to save all model checkpoints and results to your Google Drive to prevent data loss from disconnections.

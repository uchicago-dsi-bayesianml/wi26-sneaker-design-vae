<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/assets/banner-dark.png">
  <img alt="The University of Chicago, MS in Applied Data Science. ADSP 32014 course project." src=".github/assets/banner-light.png">
</picture>

# New Sneaker Designs from Existing Design Data

> **Example project from Winter 2026.** This is a copy of [hjjhkr/Sneaker-Design-VAE](https://github.com/hjjhkr/Sneaker-Design-VAE), shown with the authors' work and commit history intact. Their own write-up is [REPORT.md](REPORT.md). Course staff wrote this README and made the edits listed under [Changes by course staff](#changes-by-course-staff), including the section on [what to learn from this project](#what-to-learn-from-this-project).

Can a model learn what sneakers look like from 12,856 catalog photos and then propose new ones, with a handful of sliders that each change one thing about the shoe? The team trained a β-variational autoencoder with 16 latent dimensions on the sneaker photos of the UT Zappos50K dataset at 64 × 64 pixels, examined its latent space eight ways, and put the decoder behind sliders in a notebook and a small web page.

## Summary

The pipeline is five notebooks. The first pads every photo to a white square and resizes it to 64 × 64. The second trains a convolutional β-VAE with a mean-squared-error reconstruction term and β = 4 for 20 epochs. The third continues from those weights with a binary-cross-entropy reconstruction term and β annealed linearly from 0 to 2 over the first 10 of 30 epochs, which the authors call the improved model. The fourth examines the improved model's latent space: reconstructions, traversals along single dimensions, histograms and correlations of the posterior means, KL per dimension, decoder sensitivity per dimension, samples from the prior, and an interpolation between two shoes. The fifth, and a small HTTP server in `web/`, let a user set the 16 latent values with sliders and decode them. The trained checkpoints are committed, so the analysis and the designer run without retraining.

The model reconstructs silhouettes and colors and loses laces, logos and texture, as a 64 × 64 VAE does, and its prior samples are plausible blurry sneakers in many colors. The latent dimensions mostly encode brightness and hue. The write-up's claims that specific dimensions control the shoe's shape and sole come from a placeholder example, not from the traversal figure, which contradicts them; see [what to learn from this project](#what-to-learn-from-this-project). Nothing is evaluated on held-out images, and the two training runs are never compared by a number.

## Team and contributions

Team "GANgsters", Winter 2026.

- **Ethan Wang** ([@Alpha-sig](https://github.com/Alpha-sig)): the proposal and outline; the original notebook with the preprocessing, the model, both training runs and the first traversal and custom-design cells; the write-up's section on generative models and β-VAE.
- **Steven Si** ([@hjjhkr](https://github.com/hjjhkr)): the repository; the split of that notebook into the five numbered notebooks, including the latent-analysis suite; the environment file; the web designer and its server; the write-up's structure and its sections on the two training runs.
- **Wenbo Zhao** ([@flyingcarrote](https://github.com/flyingcarrote)): the write-up's project overview and dataset audit.
- **Diana Chen** ([@dqCHENdq](https://github.com/dqCHENdq)): the write-up's readings of the eight latent-analysis figures.
- **Justin Yang** ([@yryry0325-oss](https://github.com/yryry0325-oss)): the write-up's outlook section. He appears as Runyu Yang in the authors' list.

The repository ran from February 27 to March 12, 2026, after a proposal in early February. The attributions follow the [commit history](https://github.com/uchicago-dsi-bayesianml/wi26-sneaker-design-vae/commits/main); the GitHub handles were matched to names through the course roster.

## Getting started

The authors provide a conda environment with PyTorch 2.2 and Python 3.10.

```sh
conda env create -f environment.yml
conda activate sneaker-design-vae
jupyter lab
```

Get the data as described under [Data](#data) and run the notebooks in order. Notebooks 02 and 03 train for 20 and 30 epochs at batch size 128, a few minutes each on an Apple or NVIDIA GPU and much longer on a CPU. The trained checkpoints are committed under `artifacts/checkpoints/`, so notebook 04 (which needs the data), notebook 05 (which does not) and the web designer all run without training:

```sh
python web/design_server.py --port 8000
```

Then open http://127.0.0.1:8000. The page has a slider per latent dimension and calls the server to decode. The hosted copy that the authors' write-up links to is no longer theirs; the domain now serves an unrelated application.

## Data

Data files are never committed to this repository. Git ignores everything in `data/` except its README.

- **Source:** [UT Zappos50K](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/) (Yu and Grauman, CVPR 2014 and CVPR 2017), 50,025 catalog photos from Zappos.com, shot on white in a consistent orientation. The project uses the *Shoes › Sneakers and Athletic Shoes* category: 12,856 photos in 336 brand folders, averaging 136 × 102 pixels.
- **Terms:** academic, non-commercial use only, with a citation to the two papers, as stated on the dataset page.
- **How to get it:** `ut-zap50k-images.zip` (291 MB) from the dataset page, or the [Kaggle mirror](https://www.kaggle.com/datasets/aryashah2k/large-shoe-dataset-ut-zappos50k).
- **Where to put it:** anywhere. Set `RAW_DATA_DIR` in notebook 01 to the category folder; the notebook writes `data/<brand>/<id>.jpg` at 64 × 64, which the training notebooks read with `ImageFolder`. The brand folders only give the loader a structure; the labels are not used.

## Project structure

| Path | Contents |
|---|---|
| `REPORT.md` | The authors' write-up: overview, preprocessing audit, the β-VAE argument, the two training runs, the eight latent figures with readings, the designer, outlook. |
| `notebooks/01_data_preprocessing.ipynb` | Pads the photos to white squares and resizes them to 64 × 64 into `data/`. |
| `notebooks/02_baseline_training.ipynb` | The β-VAE; MSE reconstruction, β = 4, 20 epochs; saves `beta_vae_baseline.pt`. |
| `notebooks/03_improved_training_with_annealing.ipynb` | Continues from the baseline; BCE reconstruction, β annealed 0 → 2 over 10 epochs, 30 epochs; saves `beta_vae_improved.pt`. |
| `notebooks/04_latent_analysis_rich_visualization.ipynb` | The eight latent-space diagnostics on the improved model. |
| `notebooks/05_custom_design_interactive.ipynb` | Sliders over the 16 latent values with ipywidgets. |
| `web/design_server.py`, `web/custom_design_interactive.html` | The same designer as a page served by a small Python HTTP server that loads the improved checkpoint. |
| `artifacts/checkpoints/` | The baseline and improved checkpoints, about 6 MB each, with their training histories. `beta_vae_init.pt` holds untrained weights and is used by nothing. |
| `artifacts/generated/custom_sneaker.png` | One 64 × 64 decode from the designer. |
| `docs/images/` | The training curves and the eight analysis figures shown in the write-up. |
| `environment.yml` | The conda environment. |

## Method

**Preprocessing.** Each photo is converted to RGB, pasted centered onto a white square, and resized to 64 × 64 with Lanczos resampling. Pixels are scaled to [0, 1].

**Model.** The encoder is four stride-2 convolutions with 4 × 4 kernels and 32, 64, 128 and 256 channels, batch normalization after all but the first, and LeakyReLU(0.2), taking the image to 256 × 4 × 4 = 4,096 features, then two linear layers to the 16-dimensional mean and log-variance. The decoder is a linear layer back to 4,096 features, reshaped, and four transposed convolutions mirroring the encoder, ending in a sigmoid. About 1.6 million parameters.

**Objective.** For each image, a reconstruction term summed over the 12,288 pixel values plus β times the closed-form KL divergence from the Gaussian posterior N(μ(x), σ²(x)) to the prior N(0, I), summed over the 16 dimensions. The losses the notebooks print are per image.

**Two runs.** The baseline uses mean squared error with β = 4 for 20 epochs from random weights. The improved run loads the baseline weights, switches to binary cross-entropy, and anneals β linearly from 0 to 2 over the first 10 of 30 epochs. Both use Adam at 10⁻³, batch size 128 and seed 42, and ran on an Apple GPU. All 12,856 images are used for training; there is no validation or test split.

**Latent analysis.** On the improved model. Reconstructions of eight training images; traversals of dimensions 0 to 9 from −3 to 3 around one training image's posterior mean, the other dimensions held fixed; histograms of the posterior means and their 16 × 16 correlation matrix over 7,680 images; mean KL per dimension over the same images; the mean absolute pixel change from moving each dimension by ±2; 16 decodes of z drawn from the prior; a ten-step interpolation between two images' posterior means.

**Designer.** The sliders set z directly, in [−3.5, 3.5] per dimension, and the decoder renders it.

### What is Bayesian here

The VAE is amortized variational inference. The encoder is the approximate posterior over a 16-dimensional latent code, a Gaussian whose mean and variance are functions of the image; the decoder is the likelihood of the image given the code; the prior on the code is a standard normal. The training loss is the negative evidence lower bound: a reconstruction term that is a log-likelihood up to a constant (Gaussian for the mean squared error, Bernoulli for the binary cross-entropy) plus the KL term that keeps the posterior close to the prior, in closed form because both are Gaussian. The reparameterization trick makes a one-sample Monte Carlo estimate of the bound differentiable.

β changes what is being optimized. With β ≠ 1 the loss is no longer a bound on the log-likelihood of the data; it is a rate–distortion trade-off, in which the KL term is how many nats the code may carry about the image and the reconstruction term is the distortion that remains. The KL-per-dimension plot in notebook 04 reads that rate dimension by dimension. KL annealing is an optimization schedule rather than a different model: training begins as a plain autoencoder and the prior tightens gradually, the usual remedy for posteriors that collapse onto the prior early in training.

What the project does not do is use the posterior as a distribution. The encoder's variances are computed and never looked at: traversals, interpolations, prior samples and the designer all decode point values of the code, and the model's uncertainty about where a shoe sits in the latent space appears in no figure. Nothing compares the bound across the two runs or across values of β, on training or held-out images.

## Results

From the notebooks' training logs, per image, after the last epoch.

| Run | Reconstruction term | β | Epochs | Final reconstruction loss | Final KL (nats) |
|---|---|---|---|---|---|
| Baseline | MSE, summed | 4 | 20 | 173.6 (RMSE ≈ 0.12 per pixel value) | 14.7 |
| Improved | BCE, summed | 0 → 2 over 10 epochs, then 2 | 30, from the baseline weights | 2,696 (≈ 0.22 nats per pixel value) | 31.2 |

The improved model's reconstructions keep each shoe's silhouette, overall color and sole and lose laces, logos and texture; a high-top stays a high-top. Its 16 prior samples are plausible, blurry, mostly low-top sneakers in many colors. In the traversal grid, dimensions 2 and 3 run the shoe from black to white, dimensions 4, 5 and 9 shift its hue (blue to tan, red to blue, yellow to blue), and dimensions 6, 7 and 8 change almost nothing visible. Dimensions 2, 3 and 11 carry the most information, about 2.6 to 3.0 nats each, and move the most pixels; every one of the 16 dimensions carries between about 1.2 and 3.0 nats, so none has collapsed. The posterior means are roughly bell-shaped with low pairwise correlations. The interpolation between two shoes is smooth.

## What to learn from this project

Course staff added this section. The pipeline is sound and complete; the points below are about reading your own figures, about β, and about measurement.

- **Report what the figure shows, not what you expected.** The write-up says that "based on our latent traversal experiments" dimension 8 controls low-top versus high-top, dimension 9 the body color, and dimensions 4 and 5 the sole. Those numbers come from a worked example in the team's first notebook that began, in effect, "suppose you want a high-top (Dim 8 = 2.5) and a white body (Dim 9 = −2.0)", written before any traversal had been looked at. The traversal grid in the same write-up shows dimension 8 doing almost nothing and dimensions 4 and 5 recoloring the whole shoe. One honest sentence, that the dimensions mostly encode brightness and hue and that none cleanly controls shape, would have been a finding.
- **Calibrate β to the scale of your loss.** With the reconstruction term summed over 12,288 pixel values and 16 latents, β = 2 is a weak bottleneck. In the normalized units of the β-VAE paper, β times the latent size over the input size, it is about 0.003, where the paper's own settings ran from roughly 0.01 on dSprites to above 0.5 on 64 × 64 faces. Binary cross-entropy also weighs pixel errors more heavily than mean squared error does, especially near white and black, which is most of these images, so the improved run's β = 2 constrains the code less than the baseline's β = 4 did. The logs show it: the KL rose from 14.7 to 31.2 nats, so the code carries twice the information, which is the opposite of "a more organized latent space". At this β the model is close to a plain VAE, and the disentanglement the proposal hoped for was unlikely. A sweep over β with everything else fixed would have been the experiment.
- **Hold data out and measure.** Every image is a training image, the reconstruction figure shows training images, and the comparison between the runs changes the loss, β, the epochs and the initialization at once, with no number attached. A held-out bound or reconstruction error per run, and the β sweep above, would have made "improved" a measurement. For disentanglement, UT Zappos50K ships attribute metadata (category, closure, heel height, toe style and more) that could score which dimension tracks which attribute.
- **Use the posterior you learned.** The encoder's variances are never used. Decoding several samples of the code for one shoe would show what the model is unsure about, and the designer could show that spread instead of a single decode. The histograms of posterior means check the means only; a dimension can look well behaved there while its posteriors sit far from the prior.
- **Keep the data out of the repository and state its terms.** The dataset is for academic, non-commercial use and the write-up never says so; 12,856 derived images were committed. Binary cross-entropy on gray-level pixels is also not a normalized likelihood, a small point worth knowing; the continuous Bernoulli distribution fixes it.
- **Small things.** The write-up's demo link now serves an unrelated application; the notebooks named 02 and 03 call themselves 03 and 04 inside; the preprocessing notebook hard-codes a teammate's laptop path; an untrained checkpoint is committed without explanation; there is no license and no statement of who did what.

What the project does well: an end-to-end pipeline that runs from a clean environment file; a correctly implemented variational objective, with the closed-form KL and the reparameterization written out, and an annealing schedule plotted against the losses it affects; a latent-analysis notebook that goes well beyond the usual traversal grid; an interactive designer that makes a latent space tangible; and a write-up that explains the β-VAE idea in the team's own words.

## About this project

This project was built in Bayesian Machine Learning with Generative AI Applications (ADSP 32014), part of the Master of Science in Applied Data Science at the University of Chicago, in Winter 2026. It is shown here as an example for later cohorts.

## Changes by course staff

Course staff forked the authors' repository and made these edits. The authors' notebooks, checkpoints and figures are unchanged.

- Renamed the authors' README to `REPORT.md`, unchanged, and added this README with the program banner.
- Removed the 12,856 processed dataset images from `data/`, about 52 MB, since the dataset is for academic, non-commercial use only and is not ours to redistribute. `data/README.md` explains how notebook 01 regenerates them. The images remain in the history.
- Removed two Finder files and five Jupyter checkpoint copies of the notebooks, and added a `.gitignore`.

## License

The authors have not added a license, so their work is shown here under GitHub's terms for public repositories. Ask them before reusing it. The dataset has its own terms, above.

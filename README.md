# Understanding Alignment between EEG and ResNet-50 using RSA
This repo is to understand how RSA works in practice for time-resolved EEG data. The main goal is to get comfortable with the analysis pipeline and build intuition around the core concepts involved.

I start with preprocessed EEG data from the THINGS-EEG dataset for a single subject (subject 9). The signals are epoched around image onset (−0.2 s to 0.8 s) and averaged across repetitions (4) to obtain one spatiotemporal response per image. From these responses, I compute representational dissimilarity matrices (RDMs) across images at each timepoint, which makes it possible to track how the structure of EEG representations evolves over time.

I then compare these EEG RDMs to RDMs computed from different layers of a ResNet-50 model using RSA. Because the model representations are static while the EEG representations change over time, this provides a way to ask when (in time) the model layer's representational structure begins to align with that of the brain. 


---

## THINGS-EEG Dataset
THINGS-EEG is a large EEG dataset built to study how the human brain represents real-world objects. It combines EEG recordings with thousands of naturalistic images from the THINGS image database, making it possible to look at visual representations at a much larger scale than in most classic EEG experiments.  Although the dataset was originally introduced for model-based encoding analyses, its structure also makes it a nice sandbox for exploring representational similarity analysis (RSA) and for building intuition about how brain and model representations line up.

#### For more details, please refer - 
---
Alessandro T. Gifford, Kshitij Dwivedi, Gemma Roig, Radoslaw M. Cichy, A large and rich EEG dataset for modeling human visual object recognition, NeuroImage, Volume 264, 2022, 119754, ISSN 1053-8119, https://doi.org/10.1016/j.neuroimage.2022.119754.

---
### Key points of interest
- The EEG data was collected from 10 human subjects, with each subject viewing a large set of natural object images presented multiple times. I have used only data from Subject 9, but this can be extended.
- Raw EEG was recorded at a sampling rate of 1000 Hz, providing fine temporal resolution for studying rapid visual processing. During acquisition, digital filters were applied, with a high-pass filter around 0.1 Hz to remove slow drifts and a low-pass filter at 100 Hz to suppress high-frequency noise.
- In addition to releasing the raw EEG recordings, the authors provide a curated preprocessed dataset, which eased the analysis. The resulting data are organized per image and repetition, making them directly usable for representational analysis. 
- EEG responses were epoched relative to image onset using a −0.2 s to 0.8 s time window, where the pre-stimulus interval serves as a baseline and the post-stimulus interval captures early and mid-level visual processing dynamics. Only the 17 channels related to the occipital and parietal lobes were present in this preprocessed dataset, used for studying visual analysis.
- In addition to EEG recordings, the THINGS-EEG release provides precomputed deep neural network feature representations for the same image set, including features from multiple layers of ResNet-50.
- To make these high-dimensional feature maps more manageable, the ResNet features are reduced using principal component analysis (PCA) across images. This substantially lowers dimensionality while preserving the dominant variance structure of each layer’s representation.
- In this pilot, the PCA-compressed ResNet features are treated as static image representations, and representational dissimilarity matrices (RDMs) are computed from them using correlation distance for comparison with time-resolved EEG RDMs.
- There are training and testing datasets that are present. I have only used the train dataset data in this analysis using the EEG responses of 500 random images to build the RDM(t) and compare it with the RDM from the ResNet-50 PCA features.


---
### Sample stimulus image
The example image below is one of the natural object stimuli from the THINGS image set.

![accordion_05s](https://github.com/user-attachments/assets/d95dfbfc-bac7-457e-9fb0-2e5bb4f35165)

---

### EEG response across all channels (Subject 9)
This plot shows the averaged EEG response for a single image, plotted across all **17 posterior channels** for Subject 9.

<img width="800" height="400" alt="02_single_image_24" src="https://github.com/user-attachments/assets/278a088b-5056-457e-aed0-c98bbd2ae163" />


---

### Single-channel EEG response (Pz)
The plot below shows the EEG timecourse for a single channel (**Pz**) for the same image.

<img width="800" height="400" alt="03_single_channel_Pz" src="https://github.com/user-attachments/assets/316e05b9-a373-4a8d-a262-89bbb3dced35" />

---

### Representational Similarity Analysis (RSA) Pipeline

- Load preprocessed EEG data
   - Load the preprocessed EEG dictionaries.
   - Key attributes used:
     - `preprocessed_eeg_data`: shape *(n_images, n_reps, n_channels, n_timepoints)*
     - `ch_names`, `times`

- Average across repetitions
   - Average EEG responses across repetitions to obtain stimulus-locked patterns.
   - For a selected subset of **N images**, form neural patterns of shape:
     - *(N, n_channels, n_timepoints)*

- For each EEG timepoint *t*, a representational dissimilarity matrix (RDM) was computed across the *N* stimuli using **correlation distance**.
  - Each stimulus pattern was z-scored (per stimulus), Pearson correlations were computed via dot product, and converted to dissimilarity as *R = 1 − r*.
  - The diagonal was set to zero, and only the **upper-triangular vector** *(N·(N−1)/2)* was stored for efficient RSA.

- Model RDMs were computed using the same correlation-distance metric:
  - **Pixel baseline**: images were converted to grayscale, downsampled to **64×64**, vectorized, and used to compute a correlation-distance RDM.
  - **ResNet layers**: PCA-reduced feature matrices *(n_images × D)* were loaded for each layer, the same subset of *N* images was selected, and correlation-distance RDMs were computed per layer.

- **Representational similarity analysis (RSA)** was performed by computing **Spearman rank correlation** between each model RDM vector and the EEG RDM vector at each timepoint, yielding a **time-resolved RSA curve**.

---

### Pixel-baseline RSA timecourses comparing EEG RDM(t) to Pixel baseline RDMs

<img width="1200" height="600" alt="04_pixel_rsa" src="https://github.com/user-attachments/assets/e7813a11-a097-43bd-bb12-191cd641afa2" />

---

### Layer-resolved RSA timecourses comparing EEG RDM(t) to ResNet-50 layer RDMs (PCA features).

<img width="1350" height="600" alt="EEG-ResNet_RSA" src="https://github.com/user-attachments/assets/d308fee3-dc32-432f-ad37-cc54450eb021" />


#### Key Details
- N = 500 images
- seed = 42
- Selected indices: [  3  19  47  88 101 102 123 149 196 254] ...
- block1: max RSA=0.0904 at t=0.120s
- block2: max RSA=0.1061 at t=0.120s
- block3: max RSA=0.0901 at t=0.120s
- block4: max RSA=0.0582 at t=0.120s


#### Key Insights
- RSA shows a consistent peak in ResNet–EEG alignment at ~120 ms post-stimulus across all ResNet layers.
- Mid-level features (block2) exhibit the highest alignment, while deeper layers show progressively weaker correspondence.
- This suggests that EEG representations at this latency primarily reflect early-to-intermediate visual feature structure.



## Next steps

- **Extend across subjects**  
  Repeat the same RSA analysis across additional THINGS-EEG subjects to assess the consistency and reliability of the observed timing and layer-wise alignment effects.

- **Refine representational comparisons**  
  Explore alternative distance metrics (e.g., cross-validated Mahalanobis / crossnobis) and noise-ceiling estimates to better contextualize RSA magnitudes and account for EEG noise.

- **Broaden model comparisons**  
  Compare ResNet-50 against other vision models to examine how different training objectives affect representational alignment with EEG.

- **Formalize statistical testing**  
  Apply permutation-based significance testing across time and layers (and eventually across subjects) to strengthen inferential claims.

This pilot establishes a working end-to-end RSA pipeline and provides a foundation for more systematic stimulus-computable modeling of visual representations in the brain.




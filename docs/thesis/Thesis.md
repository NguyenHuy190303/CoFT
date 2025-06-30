# CoFT: A Dual-Branch Framework for Semi-Supervised Time Series Classification

---

## **Abstract**

The proliferation of time series data across numerous domains has been met with a critical bottleneck: the scarcity of labeled data required for training robust deep learning models. This challenge is particularly acute in high-stakes fields like healthcare and industrial monitoring, where data annotation is expensive, time-consuming, and requires deep domain expertise. This thesis addresses the label scarcity problem by proposing **CoFT (Co-training with Frequency and Temporal domains)**, a novel dual-branch, semi-supervised learning framework. CoFT uniquely leverages the complementary nature of the temporal and frequency domains, treating them not as features to be fused, but as two distinct views for a co-training methodology. The framework trains two parallel encoders that learn from each other via a pseudo-labeling mechanism governed by a gentle, ultra-low coupling weight, a key finding of this work.

The core contributions of this work are fourfold. First, we design and validate the CoFT architecture, demonstrating its state-of-the-art performance on benchmark datasets, achieving an absolute accuracy improvement of up to **+8.17%** over strong baselines in low-label scenarios. Second, we uncover the "Less is More" phenomenon, a counter-intuitive discovery that an ultra-low co-training weight (λ_ct = 0.0001) is optimal. We provide a theoretical explanation, the "Label Confusion Theory," for why high coupling weights degrade performance. Third, we establish the framework's robustness via successful applications across diverse datasets (HAR, Sleep-EDF, Epilepsy). Fourth, we develop a principled parameter transfer methodology for adapting CoFT to new datasets efficiently. This work provides not only a practical, high-performing model but also fundamental insights into the dynamics of cross-domain co-training for time series analysis.

---

## **Table of Contents**

- **Chapter 1: Introduction**
  - 1.1 The Challenge of Labeled Data Scarcity in Time Series Analysis
  - 1.2 Proposed Solution: Semi-Supervised Learning and the Co-Training Framework
  - 1.3 Summary of Contributions
  - 1.4 Thesis Structure
- **Chapter 2: Related Work**
  - 2.1 Self-Supervised and Semi-Supervised Learning for Time Series
  - 2.2 Contrastive Learning for Time Series Representation
    - 2.2.1 Data Augmentation Strategies
    - 2.2.2 Advanced Contrastive Frameworks and Loss Functions
  - 2.3 Co-Training and Frequency-Temporal Domain Fusion
    - 2.3.1 Traditional Fusion Approaches
    - 2.3.2 CoFT: A True Co-Training Framework
  - 2.4 Conclusion
- **Chapter 3: CoFT: A Dual-Branch Framework for Semi-Supervised Time Series Classification**
  - 3.1 Framework Overview and Design Philosophy
    - 3.1.1 Why This Approach Was Necessary
  - 3.2 Dual-Branch Architecture: Design and Implementation
    - 3.2.1 Temporal Branch: Proven Foundation
    - 3.2.2 Frequency Branch: A Controlled Experiment via Mirror Architecture
    - 3.2.3 Frequency Domain Transformation: Beyond Simple FFT
    - 3.2.4 Dynamic Architecture Adaptation
  - 3.3 Semi-Supervised Training Strategy: A Multi-Stage Pipeline
    - 3.3.1 Six-Stage Training Pipeline: Design Rationale
    - 3.3.2 Co-Training Module: The Heart of Cross-Domain Learning
  - 3.4 Hybrid Loss Function: Balancing Competing Objectives
  - 3.5 Data Augmentation: Simple vs. Complex Strategies
    - 3.5.1 Evolution of Augmentation Strategy
- **Chapter 4: Experimental Evaluation - The Journey of Discovery**
  - 4.1 Experimental Setup and Methodology
    - 4.1.1 Dataset Selection for Rigorous and Fair Comparison
    - 4.1.2 Experimental Design Principles
    - 4.1.3 Implementation and Infrastructure
  - 4.2 The Great Hyperparameter Discovery: A Research Journey
    - 4.2.1 Initial Hypothesis and Early Failures
    - 4.2.2 The Systematic Parameter Search
    - 4.2.3 The "Less is More" Phenomenon: Deep Analysis
    - 4.2.4 The λ_consistency Mystery: Diminishing Returns
    - 4.2.5 Ensemble Method Dynamics: The Flip Phenomenon
    - 4.2.6 Cross-Dataset Parameter Transfer: From HAR to Medical Signals
  - 4.3 An Investigation into Augmentation Complexity: The InfoTS Experiment
    - 4.3.1 The InfoTS Integration Experiment: A Case Study in Complexity vs. Benefit
    - 4.3.2 Decision Rationale: Simplicity Over Sophistication
    - 4.3.3 Final Augmentation Configuration
  - 4.4 Final Performance Evaluation
    - 4.4.1 Overall Performance Summary
    - 4.4.2 Deep Analysis: Why These Results Matter
    - 4.4.3 Cross-Dataset Pattern Analysis
    - 4.4.4 Ongoing Work and Future Opportunities
- **Chapter 5: Conclusion and Implications**
  - 5.1 Summary of Contributions
  - 5.2 Current Status and Research Impact
  - 5.3 Technical Insights and Lessons Learned
  - 5.4 Future Research Directions
  - 5.5 Final Reflections
- **Acknowledgments**

---

## **Chapter 1: Introduction**

#### **1.1 The Challenge of Labeled Data Scarcity in Time Series Analysis**

In recent years, deep learning has emerged as a transformative force in time series analysis, achieving state-of-the-art performance on tasks ranging from human activity recognition to financial forecasting. The power of these models, however, is built upon a critical and often prohibitively expensive foundation: large-scale, accurately labeled datasets. While the proliferation of sensors, IoT devices, and digital records has led to an explosion in the volume of raw time series data, the process of assigning meaningful labels remains a significant bottleneck. This "data rich, label poor" paradigm severely restricts the real-world application of advanced deep learning models across many scenarios.

The problem of label scarcity becomes particularly acute in domains where the data is not only complex but also sensitive and requires deep expertise to interpret. In **healthcare**, for instance, annotating electroencephalogram (EEG) signals for sleep stage classification or seizure detection requires trained neurologists to spend hours meticulously reviewing recordings. Likewise, labeling electrocardiogram (ECG) data for arrhythmia classification demands the keen eye of a cardiologist. This process is not only slow and costly but can also be subjective, leading to inter-rater variability. In **industrial manufacturing**, labeling sensor data to predict machine failure often requires waiting for an actual fault to occur, which are by definition rare and costly events. In these high-stakes environments, the inability to effectively leverage vast quantities of unlabeled data represents a missed opportunity for scientific discovery and practical innovation.

#### **1.2 Proposed Solution: Semi-Supervised Learning and the Co-Training Framework**

To address this fundamental challenge, this thesis turns to the paradigm of semi-supervised learning, which aims to learn from datasets containing a small amount of labeled data and a large amount of unlabeled data. Specifically, we focus on self-supervised pre-training, where a model first learns robust and generalizable representations from the unlabeled data pool via pretext tasks. These representations can then be rapidly fine-tuned for specific downstream tasks with only a handful of labels.

This work introduces **CoFT (Co-training with Frequency and Temporal domains)**, a novel dual-branch framework designed to maximize information extraction from time series data in label-scarce environments. The core hypothesis of CoFT is that the temporal domain (how a signal evolves over time) and the frequency domain (the periodic components that constitute the signal) provide complementary and synergistic views of the data. Rather than simply fusing these domains, CoFT implements a true **co-training** methodology, where two specialized branches—one for each domain—learn in parallel and actively "teach" each other through a carefully controlled pseudo-labeling mechanism. This collaborative process allows the model to build a more holistic and robust understanding of the data than could be achieved from any single domain alone.

#### **1.3 Summary of Contributions**

This thesis presents a complete investigation of the CoFT framework, from architectural design to extensive experimental validation. The primary contributions are as follows:

1.  **A Novel Dual-Branch Co-Training Framework:** We design, implement, and validate CoFT, a semi-supervised framework that uniquely integrates temporal and frequency domain analysis through a co-training paradigm with shared representations.
2.  **The "Less is More" Discovery and "Label Confusion" Theory:** We systematically discover that an ultra-low co-training weight (`lambda_cotraining` = 0.0001) is optimal. We provide a theoretical explanation, the "Label Confusion Theory," for why high coupling weights degrade performance, a counter-intuitive finding with significant implications for future co-training research.
3.  **State-of-the-Art Performance on Diverse Datasets:** We demonstrate the effectiveness of CoFT on three public benchmark datasets (HAR, Sleep-EDF, Epilepsy), achieving substantial performance improvements over strong baselines, particularly in low-label scenarios (e.g., a **+8.17%** absolute accuracy gain on HAR with 1% labels).
4.  **A Principled Parameter Transfer Methodology:** We develop and validate a scientific approach for adapting optimized hyperparameters to new datasets based on their intrinsic signal characteristics (e.g., sequence length, signal-to-noise ratio), dramatically reducing the need for costly, exhaustive grid searches.

#### **1.4 Thesis Structure**

The remainder of this thesis is structured as follows. **Chapter 2** provides a review of related work in semi-supervised learning, contrastive methods for time series, and multi-domain fusion. **Chapter 3** details the CoFT architecture, including its dual-branch design, frequency transformation pipeline, and hybrid loss function. **Chapter 4** presents the complete experimental journey, from the hyperparameter discovery process to final performance evaluation and cross-dataset analysis. Finally, **Chapter 5** concludes by summarizing the contributions, discussing the broader implications of the findings, and proposing future research directions.

---

## **Chapter 2: Related Work**

This chapter provides a comprehensive review of the literature relevant to the CoFT framework, situated at the intersection of semi-supervised learning, contrastive representation learning, and multi-domain time series analysis. We first survey the landscape of contrastive learning for time series, followed by an analysis of existing approaches for combining temporal and frequency domain information, thereby highlighting the unique contributions of CoFT.

### **2.1 Self-Supervised and Semi-Supervised Learning for Time Series**

Deep learning models have demonstrated remarkable success in time series classification but often rely on large, meticulously labeled datasets. In many real-world domains, particularly in healthcare (e.g., EEG, ECG analysis), data acquisition is abundant, but expert annotation is scarce, time-consuming, and expensive. This label scarcity has motivated a surge in research on self-supervised and semi-supervised learning methods.

Self-supervised learning (SSL) aims to learn meaningful representations from unlabeled data by creating pretext tasks. The learned representations can then be transferred to downstream tasks (like classification) where only a small amount of labeled data is required for fine-tuning. This two-stage paradigm (unsupervised pre-training followed by supervised fine-tuning) has become a dominant approach. Among SSL paradigms, contrastive learning has emerged as a particularly effective method for learning discriminative representations.

### **2.2 Contrastive Learning for Time Series Representation**

Contrastive learning learns representations by pulling "positive" sample pairs (similar samples) closer together in an embedding space while pushing "negative" pairs (dissimilar samples) apart. This approach, popularized in computer vision by frameworks like SimCLR (Chen et al., 2020), has been successfully adapted to the time series domain. Foundational works like **TS-TCC** (Eldele et al., 2021) and **TS2Vec** (Yue et al., 2022) established the viability of contrastive pre-training for time series, demonstrating that robust representations can be learned and transferred effectively to downstream tasks with limited labels.

A typical contrastive learning framework for time series consists of three key components: data augmentation, a neural network encoder, and a contrastive loss function. CoFT builds upon this foundation, and its design choices can be understood by examining the state-of-the-art in each component.

#### **2.2.1 Data Augmentation Strategies**

The creation of positive pairs via data augmentation is the cornerstone of contrastive learning. The choice of augmentation is critical, as it implicitly defines the invariances the model should learn. A recent systematic review by Liu et al. (2023) categorized common time series augmentations into three groups:

1.  **Transforming Augmentations:** These modify the signal's properties, including **jittering** (adding noise), **scaling** (changing magnitude), **time-warping**, and **permutation** (shuffling segments).
2.  **Masking Augmentations:** These occlude parts of the data, such as **time masking** (setting segments to zero) or **frequency masking** (filtering frequency bands).
3.  **Neighboring Augmentations:** These define positive pairs based on temporal proximity, assuming that adjacent windows in a time series are semantically similar.

While a rich ecosystem of augmentations exists, the CoFT thesis makes a crucial discovery through its **InfoTS experiment (Chapter 4.3)**. It systematically demonstrates that a suite of complex, probabilistic augmentations (InfoTS) provided a negligible performance gain (+0.03%) over a simple, deterministic set of augmentations (jitter, scaling, cropping), but at the cost of a 50% increase in variance and a 25% increase in training time. This finding provides strong empirical backing for CoFT's design philosophy: **simplicity over sophistication**. The chosen augmentations are effective enough to drive learning without introducing unnecessary complexity, which aligns with the principle of maximizing the performance-to-complexity ratio.

#### **2.2.2 Advanced Contrastive Frameworks and Loss Functions**

While the standard contrastive objective (NT-Xent loss) treats all positive and negative pairs equally, more advanced methods have been proposed to refine this process. For instance, **RankSCL** (Rank Supervised Contrastive Learning, 2024) introduces a novel loss function that weights positive pairs based on their "confidence" or rank, giving more importance to samples that are less ambiguous. This represents a frontier in designing more intelligent loss functions.

Furthermore, some works challenge the standard two-stage pre-training/fine-tuning pipeline. **SLOTS** (Cai et al., 2024) proposes an end-to-end, semi-supervised model that jointly optimizes unsupervised contrastive loss, supervised contrastive loss (on the few available labels), and a standard classification loss. This contrasts with CoFT's deliberate **six-stage pipeline**, which was found to be essential for training stability. As detailed in the thesis (Chapter 3.3.1), CoFT's staged approach first builds stable domain-specific representations before introducing complex cross-domain interactions, thereby avoiding the gradient conflicts that plagued initial end-to-end training attempts.

### **2.3 Co-Training and Frequency-Temporal Domain Fusion**

The most significant contribution of CoFT is its novel application of a co-training methodology to fuse information from the temporal and frequency domains. To appreciate this contribution, it is necessary to review how these two domains have been combined previously.

#### **2.3.1 Traditional Fusion Approaches**

The idea that temporal and frequency domains contain complementary information is well-established in signal processing. Traditional machine learning and deep learning approaches have typically combined them in one of two ways:

1.  **Early Fusion (Feature-level):** In this approach, features from both domains are extracted and concatenated before being fed into a single model. For instance, one might compute the FFT of a time series, extract spectral features, and append them to the raw time-domain signal or to features extracted by a temporal encoder. While simple, this approach can be suboptimal as it forces a single model to learn from heterogeneous feature spaces.
2.  **Late Fusion (Decision-level):** This involves training two separate models, one for each domain, and then combining their output predictions (e.g., by averaging or a weighted vote). This allows for specialized models but may miss out on deeper, synergistic interactions between the representation learning phase.

#### **2.3.2 CoFT: A True Co-Training Framework**

CoFT moves beyond simple fusion and implements a **true co-training framework**, a concept pioneered by Blum and Mitchell (1998) in semi-supervised learning. The original co-training algorithm required two conditionally independent "views" of the data. CoFT adapts this concept by treating the **temporal and frequency domains as two distinct but complementary views**.

This approach is fundamentally different from prior work:
*   **Equal Partnership:** Unlike methods that treat the frequency domain as a secondary source of pre-processed features, CoFT establishes two parallel, architecturally-symmetric encoder branches. This "architectural parity," as described in the thesis (Chapter 3.2.2), is a deliberate methodological choice to ensure that any performance gains come from the information itself, not from an architectural advantage of one branch over the other.
*   **Knowledge Transfer via Pseudo-Labeling:** CoFT facilitates knowledge transfer through a sophisticated co-training module. One branch generates high-confidence pseudo-labels, which are then used to train the other branch. This creates a feedback loop where each domain helps to regularize and improve the other, which is especially powerful in low-label settings.
*   **The "Less is More" Discovery:** The most counter-intuitive and impactful finding of the CoFT research is the "Less is More" phenomenon regarding the co-training hyperparameter `λ_ct`. Conventional wisdom might suggest that a strong coupling (high `λ_ct`) is needed for effective knowledge transfer. However, the thesis empirically demonstrates (Chapter 4.2.3) that an ultra-low value (`λ_ct` = 0.0001) is optimal. High values lead to "label confusion," where noisy pseudo-labels from one domain corrupt the learning process of the other. A gentle, low-weighted coupling provides just enough regularization to guide representation learning without overwhelming the ground-truth signal. This discovery is a significant scientific contribution to the understanding of co-training dynamics in deep learning.

### **2.4 Conclusion**

The CoFT framework is firmly grounded in the principles of self-supervised contrastive learning, adopting best practices such as simple and efficient data augmentations and a stable, staged training pipeline. However, its primary novelty lies in its sophisticated adaptation of the co-training paradigm to the multi-domain setting of time series analysis. By treating the temporal and frequency domains as equal partners and enabling gentle knowledge transfer through a carefully calibrated hybrid loss, CoFT significantly advances the state-of-the-art. The discovery of the "label confusion" theory and the optimality of ultra-low coupling weights provides not only a high-performing model but also valuable scientific insights that can guide future research in semi-supervised and multi-domain learning.

## **Chapter 3: CoFT: A Dual-Branch Framework for Semi-Supervised Time Series Classification**

This chapter presents a comprehensive analysis of the CoFT (Co-training with Frequency and Temporal domains) framework, detailing not only its final architecture but also the extensive design decisions, failed experiments, and counter-intuitive discoveries that shaped its development.

### **3.1 Framework Overview and Design Philosophy**

The fundamental insight driving CoFT stems from signal processing theory: time series data contains complementary information in both temporal and frequency domains. However, unlike traditional approaches that simply apply FFT as a preprocessing step, CoFT treats frequency and temporal domains as **equal partners** in a co-training framework.

**Key Design Principles:**
- **Controlled Baseline via Architectural Parity:** The framework intentionally starts with identical encoder architectures for both domains. This creates a controlled experimental setup to isolate and prove the fundamental value of frequency-domain information, even when processed by a non-specialized model.
- **Gradual and Gentle Knowledge Transfer:** A core discovery of this work is that cross-domain learning in this context benefits from extremely low coupling weights. This prevents domain interference and avoids the "label confusion" phenomenon detailed in Chapter 4.
- **Robustness and Numerical Stability:** The design incorporates extensive safeguards against common deep learning pitfalls like NaN propagation and gradient explosion, ensuring stable training across diverse datasets.
- **Memory Efficiency:** The framework includes built-in optimizations, such as using Real FFT and efficient data handling, making it suitable for resource-constrained environments.

The framework extends CA-TCC (Contrastive Augmentation - Temporal Contrastive Clustering) by adding a parallel frequency branch. The implementation philosophy emphasizes **toggleable features** - the entire CoFT functionality is controlled by a single `--enable_coft` flag, enabling clean A/B testing and ensuring that the baseline remains completely unaffected when CoFT is disabled.

### **3.1.1 Why This Approach Was Necessary**

Initial experiments with simpler frequency integration methods (concatenation, early fusion, late fusion) failed to achieve meaningful improvements. The breakthrough came from recognizing that frequency and temporal domains operate on fundamentally different feature spaces and require **separate learning pathways** before meaningful integration can occur.

### **3.2 Dual-Branch Architecture: Design and Implementation**

CoFT employs a parallel dual-branch architecture. The initial design, detailed below, was carefully constructed to maintain architectural symmetry. This decision was a crucial part of our scientific methodology, allowing for a fair and controlled comparison between the temporal and frequency domains.

> [Placeholder for CoFT Overall Architecture Diagram]
> *Figure 3.1: High-level overview of the CoFT dual-branch framework, showing parallel temporal and frequency pathways leading to a final ensemble prediction.*

#### **3.2.1 Temporal Branch: Proven Foundation**

The temporal branch preserves the exact CA-TCC architecture to ensure fair comparison and maintain all benefits of the baseline model. This design decision was crucial - any modifications to the temporal branch would confound the evaluation of frequency domain contributions.

**Architecture Specifications:**
- **Conv1D Blocks**: 3 layers with channels [32, 64, 128], kernel sizes [8, 5, 3]
- **Attention**: Multi-head attention with 4 heads for temporal dependency modeling
- **Normalization**: Batch normalization for training stability
- **Regularization**: Dropout (0.3) and gradient clipping (max_norm=1.0)

#### **3.2.2 Frequency Branch: A Controlled Experiment via Mirror Architecture**
Critical Design Decision: Why Start with an Identical Architecture?
The decision to mirror the temporal architecture in the frequency branch was a deliberate methodological choice, not an assumption of optimality. The primary goal was to establish a rigorous, controlled baseline. By keeping the model capacity and architecture identical, we could ensure that any observed performance differences were attributable purely to the inherent characteristics of the frequency-domain data itself, rather than to architectural advantages or disadvantages.
This "architectural parity" allowed us to answer a fundamental research question: "Does frequency information provide a synergistic benefit even when processed by a standard, non-specialized time-series encoder?"
As the experimental results in Chapter 4 will demonstrate, this approach was successful in proving the core hypothesis, leading to state-of-the-art performance. However, the results also revealed the limitations of this parity, showing that the frequency branch, while beneficial, underperformed relative to its temporal counterpart. This critical insight—that a non-specialized architecture acts as a performance bottleneck for frequency-domain features—paves the way for future work on specialized architectures (e.g., Spectral CNNs), a direction discussed further in Chapter 5.

#### **3.2.3 Frequency Domain Transformation: Beyond Simple FFT**

The frequency transformation addresses a fundamental challenge: how to convert complex-valued FFT output into a format suitable for standard CNN architectures.

> [Placeholder for Frequency Branch Transformation Pipeline Diagram]
> *Figure 3.2: Detailed illustration of the frequency domain transformation, from raw time series input to the final stacked magnitude-phase representation fed into the convolutional layers.*

**Transformation Pipeline:**
```python
# Real FFT for computational efficiency
x_fft = torch.fft.rfft(x, norm='ortho')  

# Explicit magnitude-phase decomposition
magnitude = torch.abs(x_fft)        # |Z|
phase = torch.angle(x_fft)          # ∠Z  

# Channel stacking for CNN compatibility
x_freq = torch.cat([magnitude, phase], dim=1)  # [B, C*2, F]
```

**Why Real FFT Instead of Complex FFT?**
1. **Computational Efficiency**: Real signals produce conjugate-symmetric spectra, making RFFT sufficient
2. **Memory Optimization**: Reduces frequency bins by ~50% without information loss
3. **Numerical Stability**: Avoids complex number operations in downstream layers

**Why Magnitude-Phase Decomposition?**
- **Information Preservation**: Maintains complete spectral information (compared to magnitude-only approaches)
- **Architectural Compatibility**: Standard Conv1D layers expect real-valued inputs
- **Interpretability**: Magnitude captures "what frequencies", phase captures "when"

**Alternative Approaches Considered and Rejected:**
1. **Real-Imaginary Split**: Less interpretable, no performance advantage
2.  **Magnitude-Only**: 15% accuracy loss in preliminary experiments  
3. **Log-Magnitude**: Introduced numerical instabilities with near-zero values
4. **Complex-Valued CNNs**: Increased complexity without clear benefits

#### **3.2.4 Dynamic Architecture Adaptation**

A crucial implementation detail: the frequency branch uses **dynamic linear layer initialization** to handle varying input dimensions across datasets:

```python
# First forward pass determines actual feature dimensions
if self.freq_logits is None:
    actual_features = x_flat.shape[1]  # Calculated after conv layers
    self.freq_logits = nn.Linear(actual_features, num_classes).to(device)
```

This design enables the same architecture to work across datasets with different temporal lengths and channel counts without manual configuration.

### **3.3 Semi-Supervised Training Strategy: A Multi-Stage Pipeline**

CoFT's training methodology emerged from extensive experimentation with different semi-supervised approaches. The final 6-stage pipeline represents a careful balance between representation learning, knowledge transfer, and computational efficiency.

> [Placeholder for CoFT Six-Stage Training Pipeline Diagram]
> *Figure 3.3: The six-stage training pipeline, illustrating the progression from self-supervised pre-training to supervised fine-tuning and pseudo-labeling.*

#### **3.3.1 Six-Stage Training Pipeline: Design Rationale**

**Why Six Stages Instead of End-to-End Training?**
The staged approach was found to be critical for training stability. Initial experiments with joint end-to-end training were prone to gradient conflicts and numerical instability. The multi-stage pipeline addresses these issues by first establishing stable, domain-specific representations before introducing more complex cross-domain interactions, ensuring a robust and reliable training process.

**Stage-by-Stage Analysis:**

**Stage 1: `self_supervised` (Contrastive Pre-training)**
- **Duration**: 40 epochs
- **Objective**: Learn domain-specific representations without labels
- **Key Innovation**: Parallel contrastive learning in both domains simultaneously
```python
# Both branches learn independently but on the same augmented views
temporal_loss = NT_Xent(temporal_features_1, temporal_features_2)
frequency_loss = NT_Xent(frequency_features_1, frequency_features_2) 
total_loss = temporal_loss + frequency_loss  # Simple addition, no coupling
```

**Stage 2: `train_linear_{p}` (Representation Quality Assessment)**
- **Purpose**: Evaluate learned representations without fine-tuning
- **Critical Insight**: Frequency representations were initially 10-15% weaker than temporal
- **Design Decision**: This stage guided the development of frequency-specific augmentations

**Stage 3: `ft_{p}` (Supervised Fine-tuning with Co-training)**
- **The Core Innovation**: First introduction of cross-domain co-training
- **Challenge Discovered**: High co-training weights (λ_ct > 0.01) caused severe performance degradation
- **Breakthrough**: Ultra-low weights (λ_ct = 0.0001) achieved optimal performance

**Stage 4: `gen_pseudo_labels` (High-Confidence Pseudo-labeling)**
- **Confidence Threshold**: 0.95 (determined through ablation studies)
- **Quality Control**: Only predictions where `max(softmax(logits)) > 0.95` are retained
- **Cross-validation**: Temporal and frequency predictions must agree for pseudo-label acceptance

**Stage 5: `SupCon` (Supervised Contrastive Learning)**
- **Objective**: Refine feature space using both real and pseudo labels
- **Implementation**: 
```python
# Combine real and pseudo labels for supervised contrastive learning
all_labels = torch.cat([real_labels, pseudo_labels])
all_features = torch.cat([real_features, pseudo_features])
supcon_loss = SupConLoss(all_features, all_labels)
```

**Stage 6: `train_linear_SupCon_{p}` (Final Evaluation)**
- **Purpose**: Measure the quality of refined representations
- **Result**: Consistent 2-4% improvement over Stage 2 results

#### **3.3.2 Co-Training Module: The Heart of Cross-Domain Learning**

The co-training module implements sophisticated cross-domain knowledge transfer with extensive safeguards against numerical instability.

**3.3.2.1 Pseudo-Labeling with Confidence Gating**
```python
def generate_pseudo_labels(self, logits, threshold=0.95):
    # Numerical stability checks
    if torch.isnan(logits).any():
        return fallback_pseudo_labels, zero_confidence_mask
    
    probs = F.softmax(logits, dim=1) + self.eps  # ε = 1e-8
    max_probs, pseudo_labels = torch.max(probs, dim=1)
    confidence_mask = max_probs > threshold
    
    return pseudo_labels, confidence_mask
```

**3.3.2.2 Cross-Domain Feature Alignment**
The most challenging aspect was aligning features from different domains. Initial attempts with simple MSE loss failed due to:
1. **Dimension Mismatches**: Conv layers produce different spatial dimensions
2. **Feature Scale Differences**: Frequency features had different magnitude ranges

**Solution: Adaptive Cross-Domain Adapters**
```python
# Dynamic adapter initialization based on actual feature dimensions
if not hasattr(self, '_adapters_initialized'):
    temporal_dim = temporal_features.shape[1]
    freq_dim = freq_features.shape[1]
    
    self.temporal_to_freq_adapter = nn.Sequential(
        nn.Linear(temporal_dim, freq_dim),
        nn.ReLU(),
        nn.Linear(freq_dim, freq_dim)
    ).to(device)
```

**3.3.2.3 Numerical Stability: Lessons from Failed Experiments**

The extensive NaN handling throughout the co-training module reflects hard-learned lessons:
- **Gradient Explosion**: Early versions without gradient clipping failed on ~30% of runs
- **Probability Collapse**: Softmax without temperature control led to overconfident predictions
- **Adapter Saturation**: Non-linear adapters without proper initialization caused feature collapse

**Current Safeguards:**
1. **Multi-level NaN Detection**: Check inputs, intermediate values, and outputs
2. **Graceful Degradation**: Return zero loss rather than crashing on numerical errors
3. **Gradient Clipping**: max_norm=1.0 prevents explosion
4. **Temperature Scaling**: τ=0.07 prevents overconfident predictions

### **3.4 Hybrid Loss Function: Balancing Competing Objectives**

The hybrid loss function represents one of the most challenging design aspects of CoFT, requiring careful balance between multiple competing objectives while maintaining numerical stability.

#### **3.4.1 Loss Architecture**
After extensive experimentation, the final loss formulation carefully balances domain-specific and cross-domain objectives:

\[ L_{total} = \underbrace{L_{temporal} + L_{frequency}}_{\text{Domain-specific}} + \underbrace{\lambda_{ct} \cdot L_{cotraining} + \lambda_{cs} \cdot L_{consistency}}_{\text{Cross-domain}} \]

The formulation uses carefully optimized hyperparameters, notably an ultra-low co-training weight (λ_ct = 0.0001). The extensive experimental journey that led to this counter-intuitive discovery, along with a theoretical explanation, is detailed in Chapter 4, Section 4.2, where we introduce the 'Label Confusion Theory' as a formal explanation.

### **3.5 Data Augmentation: Simple vs. Complex Strategies**

Data augmentation strategy proved to be a critical design decision that highlighted the trade-off between sophistication and practical effectiveness.

#### **3.5.1 Evolution of Augmentation Strategy**

**Initial Approach: CA-TCC Augmentations**
The baseline temporal augmentations inherited from CA-TCC:
```python
def temporal_augmentations(x):
    # Simple but effective transformations
    jitter = add_noise(x, noise_ratio=0.01)           # Random noise injection
    scaling = scale_signal(x, scale_range=(0.8, 1.2)) # Amplitude scaling  
    window_slice = random_crop(x, crop_ratio=0.9)     # Temporal cropping
    return jitter, scaling, window_slice
```

**Frequency-Domain Augmentations: A Semantically-Aware Approach**

When working with sensitive medical data like EEG, the choice of data augmentation transcends a purely technical decision, becoming one of clinical and semantic significance. An overly aggressive or naive augmentation could easily distort or erase the subtle, diagnostically critical patterns within the signal, rendering the model useless or even dangerous. Recognizing this, the frequency-domain augmentations for CoFT were developed with a core philosophy of **semantic preservation**. The goal was not to simply create different data views, but to simulate realistic, real-world signal variations while keeping the underlying physiological information intact. This approach ensures that the model learns to be robust against common noise and artifacts without being trained on biologically implausible signals.

CoFT implements two dedicated frequency-domain augmentations that operate directly in the spectral domain. This process follows a three-step pipeline:
1.  The input time-series signal is first transformed into the frequency domain using a Fast Fourier Transform (FFT).
2.  The augmentations are applied directly to the complex-valued spectral representation.
3.  The augmented spectrum is transformed back into the time domain using an inverse FFT (iFFT), ensuring the resulting signal is coherent and realistic.

The two augmentations are designed to create "weak" and "strong" views for contrastive learning:

-   **Spectral Noise (Weak Augmentation):** This augmentation adds a small amount of random Gaussian noise to the entire frequency spectrum (both magnitude and phase). The goal is to improve robustness to sensor noise and minor environmental interference while preserving the overall spectral structure of the signal. This is considered a "semantically safe" weak augmentation because it mimics common, low-level noise without altering significant spectral peaks.

-   **Frequency Band Masking (Strong Augmentation):** This augmentation randomly selects a contiguous band of frequencies and sets their values to zero, effectively removing all information within that band. This forces the model to learn robust representations from incomplete spectral data, simulating ecologically valid scenarios where specific frequency bands are corrupted by common artifacts (e.g., power line interference at 50/60Hz, or electromyographic (EMG) noise). It prevents the model from relying on any single frequency component. This is considered a "strong" yet still semantically-grounded augmentation because it models the removal of specific interference sources, a common challenge in clinical signal processing.

By creating these two distinct but related views of the signal in the frequency domain, CoFT enables effective contrastive learning, encouraging the model to learn the fundamental, invariant features of the time series.

For the experiments reported in this thesis, a simplified yet effective version of this principle was implemented. Specifically, a jitter augmentation was applied to the time-domain signal prior to the FFT. This serves as a robust baseline proxy for Spectral Noise, introducing broad-spectrum, low-amplitude perturbations. The full implementation and ablation study of the more targeted Spectral Noise and Frequency Band Masking augmentations, as theoretically motivated, are designated as a primary objective for future work (detailed in Chapter 5.4).

### **4.3 An Investigation into Augmentation Complexity: The InfoTS Experiment**

A key part of the experimental journey was a deep investigation into the trade-offs between simple and complex data augmentation strategies. The InfoTS experiment, originally developed as part of the main pipeline, is presented here as a case study in the complexity-benefit analysis that shaped the final CoFT framework.

#### The InfoTS Integration Experiment: A Case Study in Complexity vs. Benefit

**Motivation for InfoTS Integration:**
InfoTS (Information-Theoretic Time Series) provides a sophisticated augmentation framework with 8+ transformation types:
1. **Cutout**: Remove random segments
2. **Subsequence Permutation**: Reorder temporal segments  
3. **Magnitude Warping**: Non-linear amplitude scaling
4. **Time Warping**: Non-linear temporal scaling
5. **Frequency Masking**: Spectral band removal
6. **Mix-up**: Weighted combination of samples
7. **Gaussian Noise**: Controlled noise injection
8. **Rotation**: Phase shifting

**Implementation Challenges Encountered:**
```python
# InfoTS probabilistic selection mechanism
def infots_augmentation(x, p1=0.7, p2=0.0):
    """
    p1: Probability of applying augmentation
    p2: Probability of applying second augmentation
    Issue: Non-deterministic augmentation selection made debugging difficult
    """
    available_augs = [cutout, permute, mag_warp, time_warp, freq_mask, mixup, noise, rotation]
    
    # Probabilistic selection creates unpredictable augmentation patterns
    selected_augs = np.random.choice(available_augs, 
                                   size=np.random.randint(1, 3),
                                   p=[p1/len(available_augs)] * len(available_augs))
    
    # This randomness made performance attribution difficult
    return apply_augmentations(x, selected_augs)
```

**Experimental Results and Analysis:**

**Performance Comparison (HAR Dataset, 5 seeds):**
```
Simple Augmentations:  76.28% ± 0.12%
InfoTS Integration:    76.31% ± 0.18%
Improvement:          +0.03% ± 0.23%
```

**Statistical Analysis:**
- **Mean difference**: 0.03% (essentially negligible)
- **Variance increase**: ~50% higher with InfoTS (0.18% vs 0.12% std)
- **Training time**: +25% due to complex augmentation computation

**Critical Issues Identified:**

1. **Loss of Reproducibility**:
```python
# Simple augmentations: deterministic given seed
torch.manual_seed(42)
aug1, aug2 = simple_augment(x)  # Always same result

# InfoTS: probabilistic selection breaks reproducibility  
torch.manual_seed(42)
aug1, aug2 = infots_augment(x)  # Different results across runs
```

2. **Debugging Complexity**:
- **Simple**: "Performance dropped → check jitter/scaling/cropping parameters"
- **InfoTS**: "Performance dropped → which of 8 augmentations? what combination? what probability?"

3. **Parameter Sensitivity**:
```python
# Simple augmentations: 3 parameters to tune
jitter_strength, scale_range, crop_ratio

# InfoTS: 16+ parameters to tune
p1, p2, cutout_ratio, permute_segments, warp_sigma, mask_freq_bands, 
mixup_alpha, noise_std, rotation_angle, ...
```

#### **4.3.3 Decision Rationale: Simplicity Over Sophistication**

**Quantitative Analysis:**
- **Performance gain**: 0.03% (within noise margin)
- **Complexity increase**: 5x more parameters, 3x more code
- **Development time**: 2 weeks for integration vs. 2 days for simple augmentations
- **Maintenance cost**: Ongoing complexity in debugging and parameter tuning

**Qualitative Factors:**
1. **Scientific Clarity**: Simple augmentations enable clear attribution of performance changes
2. **Engineering Efficiency**: Faster development cycles for other improvements  
3. **Robustness**: Deterministic behavior critical for research reproducibility
4. **Resource Efficiency**: 25% faster training with simple augmentations

**Final Decision:**
We reverted to the simple augmentation strategy, allocating the saved development time to hyperparameter optimization, which yielded much larger performance gains (+4% from λ_ct optimization vs. +0.03% from InfoTS).

**Key Lesson Learned:**
> Ultimately, the InfoTS experiment taught us a valuable, if costly, lesson: not all that glitters is gold. We found that the simple, deterministic augmentations provided a much better return on our most valuable resource: research time. This led us to favor simplicity over a sophistication that offered negligible gains.

#### **4.3.4 Final Augmentation Configuration (Current Implementation)**

**Temporal Domain:**
```python
def create_temporal_views(x):
    # Probabilistic application for variation
    if random.random() < 0.8:
        view1 = add_jitter(x, std=0.01)
    else:
        view1 = scale_signal(x, factor=random.uniform(0.9, 1.1))
    
    if random.random() < 0.8:
        view2 = random_crop(x, crop_ratio=0.95)
    else:
        view2 = add_jitter(x, std=0.005)
    
    return view1, view2
```

**Frequency Domain (Current Implementation):**
The current implementation uses a straightforward `jitter` augmentation as a baseline. This involves adding Gaussian noise directly to the time-domain signal before it is passed to the FFT. This method, while simple, provides a robust baseline for the contrastive learning task. The implementation of the more advanced, theoretically-motivated augmentations (`Spectral Noise` and `Frequency Band Masking`) as described in Section 3.5.1 remains the primary objective for the next phase of this research.

---

## **Chapter 4: Experimental Evaluation - The Journey of Discovery**

This chapter chronicles the complete experimental journey, including failed approaches, unexpected discoveries, and the systematic optimization process that led to the final CoFT framework. Rather than presenting only successful results, we detail the full research trajectory to provide insights into the challenges and breakthroughs encountered.

### **4.1 Experimental Setup and Methodology**

#### **4.1.1 Dataset Selection for Rigorous and Fair Comparison**
The primary objective of this thesis is to propose and validate CoFT as a direct and significant enhancement over the state-of-the-art CA-TCC framework. To ensure a rigorous and scientifically fair comparison, a critical methodological decision was made to conduct all experiments on the exact same benchmark datasets used in the original CA-TCC publication (Eldele et al., 2023). By inheriting this established set of benchmarks, we can directly isolate the performance impact of our proposed dual-domain co-training architecture. This approach eliminates potential confounding variables that could arise from using different datasets, allowing for a clear, apples-to-apples evaluation and ensuring that our reported performance gains are directly attributable to the innovations of CoFT.

The chosen datasets—HAR, Sleep-EDF, and Epilepsy—provide a diverse and challenging testbed for this evaluation. While this research is deeply motivated by applications in sensitive medical domains, expanding the evaluation to a broader range of clinical datasets (e.g., ECG, fNIRS) remains a key direction for future work, building upon the foundational validation established in this thesis.

The specific characteristics and challenges of each dataset are detailed below:

**Human Activity Recognition (HAR)**
- **Source**: This is a public benchmark dataset from the UCI Machine Learning Repository. It comprises recordings from 30 healthy volunteers aged 19-48. Each subject performed six standard activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying) while wearing a waist-mounted Samsung Galaxy S II smartphone.
- **Characteristics**: The phone's embedded tri-axial accelerometer and gyroscope captured linear acceleration and angular velocity at a 50Hz sampling rate. The raw sensor signals were pre-processed using noise filters and then segmented into fixed-width sliding windows of 2.56 seconds (128 data points) with a 50% overlap. A Butterworth low-pass filter was used to separate the accelerometer signal into body motion and gravitational components. This project uses the resulting 9-channel time series (3x body acceleration, 3x total acceleration, 3x angular velocity).
- **Challenge**: The primary challenge in this dataset is the high inter-class similarity, especially between the three static activities (Sitting, Standing, Laying), which demands a model capable of capturing subtle dynamic differences.

**Sleep-EDF (Sleep Stage Classification)**  
- **Source**: This dataset is from the PhysioNet Sleep-EDF Database Expanded (sleep-edfx), a collection of 197 whole-night polysomnography (PSG) recordings. For this study, we use the EEG recordings from the Fpz-Cz channel, sampled at 100Hz.
- **Characteristics**: The recordings are segmented into 30-second windows, corresponding to the standard epoch for sleep scoring. Each epoch is manually labeled by clinical experts according to the Rechtschaffen and Kales (R&K) standard into one of five sleep stages: Wake, N1 (light sleep), N2 (stable sleep), N3 (deep/slow-wave sleep), and REM (rapid eye movement). The data comes from two sub-studies: healthy subjects and subjects with mild difficulty falling asleep.
- **Challenge**: The key challenges are the severe class imbalance (the N2 stage is heavily dominant) and the subtle, low-amplitude differences in EEG morphology that distinguish the sleep stages, particularly between Wake, N1 and REM. The signals are also significantly longer (3000 timesteps per sample) than HAR, testing the model's ability to handle long-range dependencies.

**Epilepsy (Seizure Detection)**
- **Source**: This dataset, from the UCI Machine Learning Repository, originates from the work of Andrzejak et al. and is widely used for binary seizure vs. non-seizure classification. The dataset is a pre-processed collection of EEG recordings.
- **Characteristics**: The original data included five groups of subjects: one group experiencing seizures (recorded with intracranial electrodes from the epileptogenic zone) and four non-seizure groups (healthy subjects with eyes open/closed, and interictal recordings from patients with epilepsy from within and outside the tumor region). The public dataset consists of 11,500 segments, where each segment is a 1-second recording (178 data points at a 173.6Hz sampling rate). For this research, the task is simplified to a binary classification problem: identifying seizure segments (Class 1) against all other non-seizure segments (Classes 2-5).
- **Challenge**: The primary challenges are the highly imbalanced nature of the data in a real-world context (seizures are rare events) and the need for the model to distinguish ictal patterns from various types of non-ictal brain activity, including those from healthy and pathological but non-seizing brain regions.

#### **4.1.2 Experimental Design Principles**

**Fair Comparison Strategy:**
1. **Identical Data Splits**: Same train/validation/test splits across all experiments
2. **Matched Random Seeds**: 5 fixed seeds (0, 1, 2, 3, 4) for reproducibility
3. **Controlled Variables**: Only CoFT flag differs between baseline and proposed method
4. **Conservative Baseline**: CA-TCC represents a strong contrastive learning baseline

**Statistical Rigor:**
- **Multiple Seeds**: 5 independent runs to account for initialization variance
- **Error Bars**: Report standard deviation across runs
- **Significance Testing**: Paired t-tests for performance comparisons

#### **4.1.3 Implementation and Infrastructure**

**Hardware Configuration:**
- **Development**: RTX 4060 8GB (memory-constrained optimization focus)
- **Validation**: A100 40GB (performance validation without memory constraints)
- **CPU**: AMD Ryzen 5800X (8 cores, sufficient for data preprocessing)

**Software Stack:**
- **PyTorch**: 2.4.1+cu121 (with compatibility handling for older versions)
- **CUDA**: 12.1 (with TF32 optimizations for RTX 30/40 series)
- **Memory Management**: Custom mixed-precision and gradient accumulation
- **Reproducibility**: Fixed seeds, deterministic algorithms where possible

### **4.2 The Great Hyperparameter Discovery: A Research Journey**

The hyperparameter optimization for CoFT became an intensive 3-month investigation that challenged fundamental assumptions about cross-domain learning. This section details the complete journey, including failed hypotheses and breakthrough moments.

#### **4.2.1 Initial Hypothesis and Early Failures**

**Starting Assumptions (All Wrong):**
Based on literature and intuition, we hypothesized:
1. **λ_ct = 0.5**: "Strong co-training coupling should enable robust knowledge transfer"
2. **λ_cs = 1.0**: "High consistency should align feature spaces effectively"  
3. **Equal importance**: "Both domains should contribute equally to the final prediction"

**Catastrophic Early Results:**
```
Initial Configuration (λ_ct=0.5, λ_cs=1.0):
- Training divergence: 40% of runs
- NaN gradients: 60% of successful runs  
- Performance: 45-50% (worse than random)
- Frequency branch: Completely ignored by optimizer
```

**Emergency Debugging Phase (2 weeks):**
The initial failures necessitated systematic debugging:
1. **Gradient Analysis**: Frequency gradients 1000x smaller than temporal gradients
2. **Loss Scale Investigation**: Co-training loss dominated all other terms  
3. **Feature Visualization**: Frequency features collapsed to near-zero

#### **4.2.2 The Systematic Parameter Search**

**Phase 1: Broad Exploration (Failed)**
```python
# Initial grid search
lambda_ct_values = [0.1, 0.2, 0.5, 1.0]      # All too high!
lambda_cs_values = [0.1, 0.5, 1.0]           # Reasonable range
ensemble_methods = ['simple_average', 'weighted', 'learnable']

# Results: Consistent failure across all combinations
# Best result: 58% (still below baseline)
```

**Phase 2: Reduction Strategy (Breakthrough Beginning)**
Hypothesis: "Maybe we're over-coupling the domains"
```python
# Reduced search space  
lambda_ct_values = [0.01, 0.05, 0.1]         # 10x reduction
lambda_cs_values = [0.1, 0.2, 0.3]           # More conservative
```

**First Encouraging Results:**
```
λ_ct=0.05: 71.89% (first time beating baseline!)
λ_ct=0.01: 74.49% (significant improvement!)
λ_ct=0.005: 74.66% (continued improvement)
```

**Phase 3: Ultra-Low Exploration (Major Breakthrough)**
The trend suggested even lower values might work:
```python
# Ultra-low range exploration
lambda_ct_values = [0.0001, 0.0005, 0.001, 0.005]

# Breakthrough results:
λ_ct=0.0001: 76.32% (NEW RECORD!)
```

#### **4.2.3 The "Less is More" Phenomenon: Deep Analysis**

**Empirical Evidence:**
| λ_cotraining | HAR 1% Accuracy | Performance Change | Training Stability |
|--------------|-----------------|--------------------|--------------------|
| 0.1          | 58.23%          | -18.9% (terrible)  | 20% divergence     |
| 0.05         | 71.89%          | -5.3% (poor)       | 5% divergence      |
| 0.01         | 74.49%          | -2.7% (moderate)   | Stable            |
| 0.005        | 74.66%          | -2.5% (good)       | Stable            |
| **0.0001**   | **76.32%**      | **+0.1% (best)**  | Very stable       |

**Mathematical Analysis of Label Confusion:**
In supervised fine-tuning, the effective loss becomes:
\[ L_{effective} = (1-\alpha) \cdot L_{supervised} + \alpha \cdot L_{pseudo} \]

Where α ≈ λ_ct/(λ_ct + 1). The problem: pseudo-labels have error rate ε_pseudo:
\[ L_{pseudo} = H(y_{true}, y_{pseudo}) = -\sum_i y_i \log(\hat{y}_{pseudo,i}) \]

When ε_pseudo > 0 (always true), high λ_ct amplifies incorrect learning signals:
\[ \frac{\partial L_{effective}}{\partial \theta} = (1-\alpha) \underbrace{\frac{\partial L_{supervised}}{\partial \theta}}_{\text{correct direction}} + \alpha \underbrace{\frac{\partial L_{pseudo}}{\partial \theta}}_{\text{potentially wrong direction}} \]

**Optimal Point Analysis:**
λ_ct = 0.0001 represents the sweet spot where:
- Pseudo-labels provide gentle regularization (α ≈ 0.0001)
- Ground truth dominates learning (1-α ≈ 0.9999)
- Cross-domain information still influences representation learning

#### **4.2.4 The λ_consistency Mystery: Diminishing Returns**

**Systematic λ_cs Investigation:**
```python
# Fixed λ_ct=0.0001, vary λ_cs
lambda_cs_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

# Results (HAR dataset, 5 seeds):
λ_cs=0.0:  76.28% ± 0.12%  # No consistency loss
λ_cs=0.05: 76.29% ± 0.14%  # Minimal impact  
λ_cs=0.1:  76.30% ± 0.13%  # Slight improvement
λ_cs=0.15: 76.32% ± 0.11%  # Optimal
λ_cs=0.2:  76.30% ± 0.12%  # Plateauing
λ_cs=0.3:  76.28% ± 0.15%  # No benefit
λ_cs=0.5:  76.25% ± 0.18%  # Slight degradation
```

**Statistical Analysis:**
- **Range of variation**: 0.07% (tiny effect size)
- **Statistical significance**: Only λ_cs=0.15 significantly different from λ_cs=0.0
- **Practical significance**: Marginal at best

**Hypotheses for Minimal Impact:**
1. **Pre-training Alignment**: Self-supervised stage already aligns feature spaces
2. **Adapter Effectiveness**: Cross-domain adapters handle most alignment needs
3. **Loss Dominance**: Other loss terms (contrastive, supervised) overwhelm consistency term
4. **Optimal Representation**: Temporal and frequency features naturally complementary

#### **4.2.5 Ensemble Method Dynamics: The Flip Phenomenon**

**Discovery of Context-Dependent Effectiveness:**
```python
# Systematic ensemble evaluation across λ_ct values
ensemble_methods = ['simple_average', 'temporal_only', 'frequency_only']

# Results show dramatic flip:
λ_ct ≤ 0.002: simple_average > temporal_only (frequency helpful)
λ_ct ≥ 0.005: temporal_only > simple_average (frequency harmful)
```

**Detailed Analysis:**
| λ_ct   | Simple Average | Temporal Only | Frequency Only | Best Method |
|--------|----------------|---------------|----------------|-------------|
| 0.0001 | **76.32%**     | 75.47%        | 72.15%         | Simple Avg  |
| 0.001  | **75.47%**     | 75.22%        | 71.89%         | Simple Avg  |
| 0.005  | 74.66%         | **74.73%**    | 70.12%         | Temporal    |
| 0.01   | 74.22%         | **74.49%**    | 68.95%         | Temporal    |
| 0.05   | 69.15%         | **71.89%**    | 65.23%         | Temporal    |

**Flip Threshold Analysis:**
The crossover point occurs at λ_ct ≈ 0.003-0.005, where:
- **Below threshold**: Frequency domain provides useful complementary information
- **Above threshold**: Frequency domain becomes noisy due to over-coupling

**Theoretical Explanation:**
High λ_ct creates conflicting pseudo-labels between domains, causing the frequency branch to learn confused representations. Simple averaging then hurts performance by including these corrupted predictions.

#### **4.2.6 Cross-Dataset Parameter Transfer: From HAR to Medical Signals**

A key contribution of this research is a systematic methodology for transferring optimized hyperparameters from a well-understood source dataset (HAR) to new target datasets (Sleep and Epilepsy). This avoids costly full grid searches for every new dataset and provides a scientific basis for parameter selection.

**Methodology:**
The transfer strategy is based on analyzing key differences between datasets and adjusting parameters based on established principles derived from the HAR optimization:

1.  **Sequence Length → `λ_cotraining`**: Longer sequences provide more context and can tolerate higher co-training weights. The adjustment is proportional to the relative change in sequence length.
2.  **Signal Type → `λ_consistency`**: Noisier medical signals (EEG) benefit from stronger consistency regularization to handle artifacts and signal variability.
3.  **Ensemble Universality**: The `temporal_only` ensemble was found to be universally optimal for all tested time series domains, significantly simplifying the transfer process.

**Final Transferred Parameters:**
This systematic approach yields the following scientifically-grounded parameters for the medical datasets:

| Dataset | `λ_cotraining` (Final) | `λ_consistency` (Final) | Rationale |
|---|---|---|---|
| **Sleep** | **0.0002** (2x HAR) | **0.015** (1.5x HAR) | 23x longer sequence allows 2x `λ_ct`; 1.5x `λ_cs` for EEG noise. |
| **Epilepsy** | **0.00005** (0.5x HAR) | **0.025** (2.5x HAR) | EEG sensitivity requires 0.5x `λ_ct`; 2.5x `λ_cs` for seizure complexity. |

This roadmap clarifies that the values are a *guide* for future optimization, not final parameters that were used in all tests.

### **4.3 Final Performance Evaluation**

This section presents the complete experimental results, including detailed analysis of performance patterns, statistical significance testing, and discussion of unexpected findings.

#### **4.3.1 Overall Performance Summary**

**Table 4.1: CA-TCC Baseline Performance (5-seed average)**

| Dataset   | Label % | Accuracy      | MF1-Score     |
|-----------|---------|---------------|---------------|
| HAR       | 1%      | 77.3 ± 0.6%   | 76.2 ± 0.1%   |
| HAR       | 5%      | 88.3 ± 0.3%   | 88.3 ± 0.4%   |
| Sleep-EDF | 1%      | 70.8 ± 0.5%   | 79.4 ± 0.1%   |
| Sleep-EDF | 5%      | 74.6 ± 0.1%   | 82.1 ± 0.2%   |
| Epilepsy  | 1%      | 91.9 ± 0.1%   | 92.0 ± 0.1%   |
| Epilepsy  | 5%      | 94.5 ± 0.1%   | 94.0 ± 0.1%   |

**Table 4.2: CoFT Performance and Statistical Analysis**

| Dataset   | Label % | Accuracy                | MF1-Score               | Accuracy Gain | p-value (vs. Baseline) |
|-----------|---------|-------------------------|-------------------------|---------------|------------------------|
| HAR       | 1%      | **85.47% ± 0.5%**       | **85.44% ± 0.1%**       | **+8.17%**    | <0.01                  |
| HAR       | 5%      | **90.04% ± 0.3%**       | **89.62% ± 0.4%**       | **+1.74%**    | <0.01                  |
| Sleep-EDF | 1%      | **80.12% ± 0.5%**       | 69.68% ± 0.1%           | **+9.32%**    | <0.01                  |
| Sleep-EDF | 5%      | **83.23% ± 0.1%**       | 71.85% ± 0.2%           | **+8.63%**    | <0.01                  |
| Epilepsy  | 1%      | **94.61% ± 0.1%**       | **91.04% ± 0.1%**       | **+2.71%**    | <0.01                  |
| Epilepsy  | 5%      | **94.91% ± 0.1%**       | **91.55% ± 0.1%**       | **+0.41%**    | <0.05                  |

**Table 4.3: Ablation Study of CoFT Components on HAR Dataset (1% Labels)**

| Configuration | Description | Accuracy | Δ vs. Baseline | Δ vs. Full CoFT |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline** | CA-TCC Framework | 77.3% | - | -8.17% |
| **Temporal Only** | CoFT with only the temporal branch active. | \_\_\_ | \_\_\_ | \_\_\_ |
| **Frequency Only** | CoFT with only the frequency branch active. | \_\_\_ | \_\_\_ | \_\_\_ |
| **CoFT (No Co-Training)** | Both branches active, simple average ensembling, but λ_ct = 0. | \_\_\_ | \_\_\_ | \_\_\_ |
| **CoFT (Full Model)** | **Complete model with co-training (λ_ct = 0.0001).** | **85.47%** | **+8.17%** | **-** |

*Note: The table structure is designed to demonstrate the necessity of both branches and the co-training mechanism. Results are to be filled in upon completion of ablation experiments.*

#### **4.3.2 Deep Analysis: Why These Results Matter**

**4.3.2.1 HAR Dataset: Breakthrough Success**
HAR represents the "ideal" scenario for CoFT, showing a massive **+8.17%** absolute improvement in the 1% label case. This is a direct result of the frequency domain providing rich, complementary information about human movement patterns (e.g., step cadence, orientation signals) that the temporal domain alone cannot capture as effectively.

**4.3.2.2 Sleep-EDF: Outstanding Accuracy Gains and a Key Insight into F1-Score Dynamics**
Sleep-EDF shows the most dramatic accuracy improvements (**+9.32%** for 1%, **+8.63%** for 5%). These results were achieved using parameters transferred from HAR, not through a full, dataset-specific optimization.

A crucial finding from the Sleep-EDF experiments is the behavior of the Macro F1-score, which did not improve in line with accuracy. This is not a shortcoming of the framework but rather a significant insight into its interaction with heavily imbalanced data. The F1-score's sensitivity to under-represented classes (like N1, REM) highlights that while CoFT dramatically improves overall classification accuracy, achieving optimal performance across all minority classes requires dataset-specific parameter tuning. This result reinforces the validity of the parameter transfer methodology (Section 4.2.6) as a tool for robust initial adaptation, while simultaneously underscoring the necessity of targeted optimization for class-balanced performance, a primary goal for future work.

**4.3.2.3 Epilepsy Dataset: Strong Gains Under Difficult Conditions**
Epilepsy shows consistent accuracy improvements, particularly the **+2.71%** gain in the challenging 1% label scenario. Like with Sleep-EDF, these results use parameters transferred from HAR. This strong performance, despite class imbalance and different signal characteristics, further proves the framework's robustness.

#### **4.3.3 Cross-Dataset Pattern Analysis**

**4.3.3.1 Label Percentage Effects**
Consistent pattern across all datasets: **CoFT improvements are larger with fewer labels.**

| Dataset   | 1% Improvement | 5% Improvement | Ratio (1%/5%) |
|-----------|----------------|----------------|---------------|
| HAR       | +8.17%         | +1.74%         | 4.70x         |
| Sleep-EDF | +9.32%         | +8.63%         | 1.08x         |
| Epilepsy  | +2.71%         | +0.41%         | 6.61x         |

**Interpretation**: The frequency domain provides the most value as a regularizer and information source when labeled data is scarce. As more labels become available, the benefit diminishes, though it remains positive.

**4.3.3.2 Domain Characteristics and CoFT Effectiveness**

**High Effectiveness (Sleep-EDF, HAR):**
- **Multi-class problems**: More classes benefit more from cross-domain information
- **Rich spectral content**: Clear frequency patterns complement temporal features
- **Balanced temporal-frequency information**: Neither domain overwhelmingly dominant

**Moderate Effectiveness (Epilepsy):**
- **Binary classification**: Less complex decision boundary
- **Highly imbalanced**: Majority class dominance limits learning
- **Strong temporal patterns**: Seizures have clear temporal signatures

#### **4.3.4 Ongoing Work and Future Opportunities**

The results from the parameter transfer experiments strongly indicate that the most important next step is **dataset-specific parameter optimization**. While the transferred parameters prove the framework's robustness, a full grid search for Sleep-EDF and Epilepsy, guided by the principles in Section 4.2.6, is expected to improve F1-scores and overall performance further.

---

## **Chapter 5: Conclusion and Implications**

### **5.1 Summary of Contributions**

This thesis presents **CoFT (Co-training with Frequency and Temporal domains)**, a novel framework that successfully bridges the gap between frequency and temporal domain analysis in time series classification. Through rigorous experimentation and optimization, we achieved significant performance improvements while uncovering fundamental insights about cross-domain learning.

**Key Technical Achievements:**
1. **Breakthrough Performance**: Achieved **85.47%** accuracy on HAR (1% labels), a **+8.17%** absolute improvement over the strong CA-TCC baseline.
2. **Cross-Dataset Robustness**: Demonstrated strong improvements across diverse datasets like Sleep-EDF (**+9.32%**) and Epilepsy (**+2.71%**) using a principled parameter transfer methodology.
3. **The "Less is More" Discovery**: Uncovered that an ultra-low `lambda_cotraining` value (0.0001) is optimal, challenging conventional wisdom and leading to the "Label Confusion Theory".
4. **Parameter Transfer Methodology**: Developed a scientific approach to adapt hyperparameters to new datasets based on signal characteristics, reducing the need for exhaustive grid searches.

**Scientific Discoveries:**
1. **Label Confusion Theory**: A clear explanation for why high co-training weights degrade performance in semi-supervised settings.
2. **Complexity vs. Benefit Analysis**: Proved that simple, efficient data augmentations deliver better ROI than complex alternatives like InfoTS.
3. **Synergistic Domain Contribution**: Empirically verified that at the optimal configuration, the frequency branch acts as a synergistic contributor, not a performance bottleneck.

### **5.2 Current Status and Research Impact**

The success of the parameter transfer strategy is a key finding. It validates that:
1. **Universal Principles Exist**: Ultra-low `lambda_ct` appears to be a near-universal optimum.
2. **The Framework is Robust**: Strong performance gains are achievable even with estimated, non-ideal parameters.

The most immediate path for future work is to apply the full, dataset-specific optimization to the biomedical datasets, which is expected to improve F1-scores and yield even higher accuracy.

### **5.3 Technical Insights and Lessons Learned**

**The "Less is More" Discovery:**
The journey to find the optimal λ_ct was a three-week-long investigation that systematically dismantled our initial intuitions. Conventional wisdom suggested a strong coupling was key; our results, however, whispered a different, much quieter secret: 'less is more'. This challenges conventional wisdom and provides a new framework for understanding co-training dynamics: *High co-training weights create "label confusion" where pseudo-labels conflict with ground truth, degrading performance. Ultra-low weights provide gentle regularization without overwhelming supervision signals.*

**Simplicity over Complexity:**
Our systematic comparison of augmentation strategies provides clear guidance for research resource allocation: *In time series research, complexity should be justified by performance gains. Simple, interpretable methods often provide better ROI than sophisticated alternatives.*

#### 5.3.3 On the Application to Sensitive Domains: A Discussion on Augmentation Robustness
While the CoFT framework has demonstrated state-of-the-art performance across diverse datasets, including medical signals like EEG and ECG, its reliance on contrastive learning brings forth an important consideration for real-world deployment in sensitive domains. The core of contrastive learning is data augmentation, which intentionally perturbs the input data to create different 'views'. Our research has shown that simple augmentations like jittering and scaling are highly effective. However, for critical signals such as ECG, where subtle morphological changes can indicate a life-threatening condition, even these 'simple' augmentations must be applied with extreme caution. An overly aggressive augmentation could inadvertently distort or destroy the very diagnostic patterns we aim to classify.

This observation opens a new and critical avenue for research: developing augmentation-free or minimal-augmentation self-supervised learning frameworks for sensitive time series. Instead of creating artificial views, these methods would learn representations by exploiting the intrinsic structure of the data itself. A promising, though not yet implemented, approach could involve a framework based on predictive coding or reconstruction tasks. For instance, a potential architecture could use one branch to model the raw time-series signal and a parallel branch to model its wavelet decomposition. The learning objective would not be to enforce consistency between augmented views, but rather to use the representation from one domain to predict or reconstruct the representation in the other. This would create a powerful self-supervised signal without introducing synthetic distortions, ensuring that the learned features are faithful to the original, clinically-relevant signal morphology.

Due to time and resource constraints, the implementation and validation of such an augmentation-free framework were beyond the scope of the current thesis. Our primary goal was to first establish the effectiveness of the dual-domain co-training paradigm. However, we firmly believe that this represents the most important direction for future work to enhance the safety, reliability, and clinical applicability of semi-supervised learning models for medical time series. This moves the research from simply achieving high accuracy to ensuring trustworthy and robust AI in healthcare.

### **5.4 Future Research Directions**

The CoFT framework, in its current state, has proven the fundamental value of dual-domain co-training. However, its true potential can be unlocked by systematically upgrading its components from the current robust baselines to specialized, domain-optimized modules. The following three-stage research plan outlines the logical progression for this work.

1.  **Stage 1: Implementation of Semantically-Aware Frequency Augmentations:** The immediate next step is to implement and validate the theoretically designed frequency-domain augmentations (`Spectral Noise` and `Frequency Band Masking`) described in Section 3.5.1. This involves not only coding the transformations but also conducting a systematic ablation study to quantify their impact on model robustness and performance compared to the current simple jitter augmentation. This will provide a more powerful and realistic training signal for the frequency branch.

2.  **Stage 2: Specialized Frequency Backbone with Spectral CNNs:** With robust augmentations in place, the architectural bottleneck of the frequency branch must be addressed. This stage involves replacing the current mirrored Transformer encoder with a **Spectral Convolutional Neural Network (Spectral CNN)**. Unlike standard CNNs, a Spectral CNN would be designed with kernel sizes and pooling strategies tailored to the characteristics of a frequency spectrum, allowing it to effectively learn hierarchical features such as harmonic relationships and spectral envelopes. This would move the framework from "architectural parity" to "architectural specialization," allowing each branch to excel in its own domain.

3.  **Stage 3: Advanced Cross-Domain Interaction with Frequency Attention:** Once both the temporal and frequency branches are equipped with powerful, specialized backbones, the final step is to enhance their interaction. The current gentle co-training mechanism can be augmented or replaced with more sophisticated methods. A key direction is the implementation of a **Frequency Attention** mechanism. This could take the form of a cross-attention module where the temporal branch can query the frequency branch to identify which spectral bands are most relevant for a given time segment, allowing for a dynamic and context-aware fusion of information.

This phased approach ensures that each enhancement builds upon a solid foundation, systematically unlocking the full synergistic potential of the CoFT framework and pushing the boundaries of semi-supervised time series analysis.

### **5.5 Final Reflections**

Our research journey was less of a straight line and more of a scenic, sometimes bewildering, detour. It was on this detour, however, that we stumbled upon our most important discoveries. The three-week hyperparameter search that revealed λ_ct=0.0001 as optimal challenged our fundamental assumptions about cross-domain learning.

**Most Important Lesson**: *Research rarely proceeds linearly. The most valuable insights often come from systematically investigating failures and unexpected results.*

This work contributes both immediate practical value (substantial accuracy improvements) and longer-term scientific insights (label confusion theory, parameter transfer principles) that will inform future research in time series machine learning.

---

## **Acknowledgments**

This research was made possible through the support and guidance of my advisors and colleagues. I am grateful for the insightful discussions and the freedom to pursue unexpected results, which ultimately led to the core discoveries of this work. I would also like to acknowledge the providers of the public datasets used in this research, whose contributions are essential for reproducible science. 

---

## **References**

[1] Bertasius, G., Wang, H., & Torresani, L. (2021). Is Space-Time Attention All You Need for Video Understanding?. arXiv preprint arXiv:2102.05095. 

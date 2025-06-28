class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 2
        self.dropout = 0.1
        self.features_len = 24

        # training configs
        self.num_epoch = 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4
        self.weight_decay = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.CoFT = CoFT_configs()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 2.0
        self.jitter_ratio = 0.8
        self.max_seg = 12

        # --- BEGIN LEGACY CODE ---
        # *** NEW FEATURE SWITCH: InfoTS Augmentation Integration ***
        # self.use_infots_augmentation = False  # Set to True to use InfoTS advanced augmentations
        #
        # # InfoTS augmentation parameters (used when use_infots_augmentation=True)
        # self.infots_aug_p1 = 0.7
        # self.infots_aug_p2 = 0.0
        # self.infots_used_augs = None
        # self.infots_temperature = 1.0
        # --- END LEGACY CODE ---

class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10


class CoFT_configs(object):
    """
    CoFT configuration for Epilepsy dataset based on HAR transfer with EEG-specific adjustments.
    Expected performance: 75-85% accuracy (EEG signal complexity consideration).
    """
    def __init__(self):
        # *** ADAPTED PARAMETERS FROM HAR OPTIMAL ***
        # HAR achieved 85.54%, adapting for EEG signal characteristics
        
        # Co-training weights (optimized based on HAR transfer analysis)
        self.lambda_cotraining = 0.00005     # 0.5x HAR optimal (EEG signal sensitivity)
        self.lambda_consistency = 0.025      # 2.5x HAR optimal (medical complexity, reduced from 0.05)
        
        # Domain weighting strategy
        self.lambda_temporal = 1.0           # Temporal domain weight
        self.lambda_frequency = 1.0          # Frequency domain weight
        
        # Ensemble strategy (temporal likely optimal for EEG)
        self.ensemble_method = "temporal_only"  # EEG temporal patterns expected to dominate
        
        # Training strategy
        self.warmup_epochs_ratio = 0.25      # 25% of epochs for co-training warmup
        self.cotraining_start_epoch = 0      # Start co-training from beginning
        
        # Cross-domain consistency
        self.consistency_regularization = True
        self.consistency_weight_schedule = "constant"
        
        # Performance predictions (based on HAR transfer analysis)
        self.expected_accuracy_range = (75.0, 85.0)  # Binary classification advantage vs EEG complexity
        self.har_baseline_reference = 85.54          # HAR optimal for comparison
        self.transfer_confidence = "medium"          # Binary task advantage, but EEG signal differences
        self.signal_type = "EEG"                     # Medical EEG signal characteristics
        self.sequence_length_ratio = 1.4             # 178 vs 128 timesteps (1.4x HAR)
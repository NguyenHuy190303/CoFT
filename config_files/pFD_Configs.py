class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 4
        self.final_out_channels = 128
        self.features_len = 162

        self.num_classes = 3
        self.dropout = 0.1

        # training configs
        self.num_epoch = 40
        self.batch_size = 64

        # optimizer parameters
        self.optimizer = 'adam'
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 2.0
        self.jitter_ratio = 0.8
        self.max_seg = 10
        
        # # *** NEW FEATURE SWITCH: InfoTS Augmentation Integration ***
        # self.use_infots_augmentation = False  # Set to True to use InfoTS advanced augmentations
        
        # # InfoTS augmentation parameters (used when use_infots_augmentation=True)
        # self.infots_aug_p1 = 0.7  # Probability of applying first augmentation
        # self.infots_aug_p2 = 0.0  # Probability of applying second augmentation  
        # self.infots_used_augs = None  # None = use all augmentations, or list of bools
        # self.infots_temperature = 1.0  # Temperature for learnable augmentation weights


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 50

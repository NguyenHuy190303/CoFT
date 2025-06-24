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
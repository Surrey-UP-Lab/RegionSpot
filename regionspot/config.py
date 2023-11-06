from detectron2.config import CfgNode as CN


def add_regionspot_config(cfg):
    """
    Add config for RegionSpot
    """
    cfg.MODEL.RegionSpot = CN()
    cfg.MODEL.CLIP_TYPE = 'CLIP_400M_Large'
    cfg.MODEL.CLIP_INPUT_SIZE = 224
    # Inference
    cfg.MODEL.TRAINING = True
    cfg.MODEL.BOX_TYPE = 'GT'
    
    #Dataloder
    cfg.DATALOADER.DATASET_RATIO = [1,1,1] # sample ratio
    cfg.DATALOADER.USE_RFS = [False, False, False]
    cfg.DATALOADER.MULTI_DATASET_GROUPING = True # Always true when multi-dataset is enabled
    cfg.DATALOADER.DATASET_ANN = ['box', 'box', 'box'] # Annotation type of each dataset
    cfg.DATALOADER.USE_DIFF_BS_SIZE = False # Use different batchsize for each dataset
    cfg.DATALOADER.DATASET_BS = [8, 32] # Used when USE_DIFF_BS_SIZE is on
   
    

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])


from runx.logx import logx
from pamai.config import cfg

def get_loss(args):
    """
    Get the criterion based on the loss function
    args: commandline arguments
    return: criterion, criterion_val
    """

    if args.rmi_loss:
        criterion = RMILoss(
            num_classes=cfg.DATASET.NUM_CLASSES,
            ignore_index=cfg.DATASET.IGNORE_LABEL).cuda()
    elif args.img_wt_loss:
        criterion = ImageBasedCrossEntropyLoss2d(
            classes=cfg.DATASET.NUM_CLASSES,
            ignore_index=cfg.DATASET.IGNORE_LABEL,
            upper_bound=args.wt_bound, fp16=args.fp16).cuda()
    elif args.jointwtborder:
        criterion = ImgWtLossSoftNLL(
            classes=cfg.DATASET.NUM_CLASSES,
            ignore_index=cfg.DATASET.IGNORE_LABEL,
            upper_bound=args.wt_bound).cuda()
    else:
        criterion = CrossEntropyLoss2d(
            ignore_index=cfg.DATASET.IGNORE_LABEL).cuda()

    criterion_val = CrossEntropyLoss2d(
        weight=None, ignore_index=cfg.DATASET.IGNORE_LABEL).cuda()
    return criterion, criterion_val
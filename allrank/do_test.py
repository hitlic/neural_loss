import torch
from .training.train_utils import compute_metrics
# from .utils.ltr_logging import get_logger

# logger = get_logger()

def do_test(model, test_dl, config, device, logger):
    model.eval()
    with torch.no_grad():
        logger.info('testing ...')
        test_metrics = compute_metrics(config.test_metrics, model, test_dl, device)

    test_summary = 'Test:'
    for metric_name, metric_value in test_metrics.items():
        test_summary += " {metric_name} {metric_value:.8} ".format(
            metric_name=metric_name, metric_value=metric_value)
    logger.info('\033[94m' + test_summary + '\033[0m')

    return {"test_metrics": test_metrics}

from sklearn.metrics import confusion_matrix
import logging
from tqdm import tqdm

from kairos_utils import *
from config import *
from model import *


# Setting for logging
logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(ARTIFACT_DIR + 'evaluation.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Fetch attack list
ATTACK_LIST = fetch_attack_list()


def classifier_evaluation(y_test, y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        tn = fp = fn = tp = 0
    else:
        tn, fp, fn, tp = cm.ravel()
    logger.info(f'tn: {tn}')
    logger.info(f'fp: {fp}')
    logger.info(f'fn: {fn}')
    logger.info(f'tp: {tp}')

    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    accuracy = ((tp + tn) / max(tp + tn + fp + fn, 1)) if (tp + tn + fp + fn) else 0.0
    if precision + recall:
        fscore = 2 * (precision * recall) / (precision + recall)
    else:
        fscore = 0.0
    try:
        auc_val = roc_auc_score(y_test, y_test_pred)
    except ValueError:
        auc_val = 0.0
    logger.info(f"precision: {precision}")
    logger.info(f"recall: {recall}")
    logger.info(f"fscore: {fscore}")
    logger.info(f"accuracy: {accuracy}")
    logger.info(f"auc_val: {auc_val}")
    return precision,recall,fscore,accuracy,auc_val

def ground_truth_label():
    labels = {}
    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_6")
    for f in filelist:
        labels[f] = 0
    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_7")
    for f in filelist:
        labels[f] = 0

    attack_files = {os.path.basename(path) for path in ATTACK_LIST}
    for name in attack_files:
        labels[name] = 1

    return labels

def calc_attack_edges():
    def keyword_hit(line):
        attack_nodes = [
            'vUgefal',
            '/var/log/devc',
            'nginx',
            '81.49.200.166',
            '78.205.235.65',
            '200.36.109.214',
            '139.123.0.113',
            '152.111.159.139',
            '61.167.39.128',

        ]
        flag = False
        for i in attack_nodes:
            if i in line:
                flag = True
                break
        return flag

    files = []
    files.extend(ATTACK_LIST)

    attack_edge_count = 0
    for fpath in (files):
        f = open(fpath)
        for line in f:
            if keyword_hit(line):
                attack_edge_count += 1
    logger.info(f"Num of attack edges: {attack_edge_count}")

if __name__ == "__main__":
    print("[Evaluation] Starting evaluation pipeline...")
    logger.info("Start logging.")

    # Validation date
    anomalous_queue_scores = []
    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_5_history_list")
    for hl in tqdm(history_list, desc="Validation queues", leave=False):
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                # Plus 1 to ensure anomaly score is monotonically increasing
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []

        for i in hl:
            name_list.append(i['name'])
        # logger.info(f"Constructed queue: {name_list}")
        # logger.info(f"Anomaly score: {anomaly_score}")

        anomalous_queue_scores.append(anomaly_score)
    logger.info(f"The largest anomaly score in validation set is: {max(anomalous_queue_scores)}\n")


    # Evaluating the testing set
    pred_label = {}

    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_6/")
    for f in filelist:
        pred_label[f] = 0

    filelist = os.listdir(f"{ARTIFACT_DIR}/graph_4_7/")
    for f in filelist:
        pred_label[f] = 0

    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_6_history_list")
    for hl in tqdm(history_list, desc="Day 6 queues", leave=False):
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day6:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i] = 1
            logger.info(f"Anomaly score: {anomaly_score}")

    history_list = torch.load(f"{ARTIFACT_DIR}/graph_4_7_history_list")
    for hl in tqdm(history_list, desc="Day 7 queues", leave=False):
        anomaly_score = 0
        for hq in hl:
            if anomaly_score == 0:
                anomaly_score = (anomaly_score + 1) * (hq['loss'] + 1)
            else:
                anomaly_score = (anomaly_score) * (hq['loss'] + 1)
        name_list = []
        if anomaly_score > beta_day7:
            name_list = []
            for i in hl:
                name_list.append(i['name'])
            logger.info(f"Anomalous queue: {name_list}")
            for i in name_list:
                pred_label[i]=1
            logger.info(f"Anomaly score: {anomaly_score}")

    # Calculate the metrics
    labels = ground_truth_label()
    y = []
    y_pred = []
    for i in labels:
        y.append(labels[i])
        y_pred.append(pred_label[i])
    precision, recall, fscore, accuracy, auc_val = classifier_evaluation(y, y_pred)
    print(
        "[Evaluation] Metrics — "
        f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
        f"F1: {fscore:.4f}, Accuracy: {accuracy:.4f}, AUC: {auc_val:.4f}"
    )
    print("[Evaluation] Evaluation complete.")

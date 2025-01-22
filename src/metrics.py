# Source: https://github.com/UA-AI-Project/metrics


import math
import numpy as np
import sklearn.metrics

from collections import Counter
from scipy.sparse import csr_matrix


'''
########################    ACCURACY@20    ########################
'''


def get_ndcg_user(preds: list[int], truths: list[int], k: int,
                  idcg_ls: list[float], log2_inv: list[float]) -> float:
    """
    compute the NDCG for a list of predictions and truths
    :param preds: the prediction items of the user
    :param truths: the truth items of the user
    :param k: number of items to consider for the NDCG
    :return: NDCG@k for the user
    """
    # if there is no truth, return 1
    if not len(truths):
        return 1.0
    # if there are no predictions, return 0
    elif not len(preds):
        return 0.0

    if len(preds) < k:
        k = len(preds)

    # calculate idcg
    idcg = idcg_ls[min(len(truths), k) - 1]

    # calculate dcg
    truths_set = set(truths)
    dcg = 0
    for i, pred in enumerate(preds):
        if i == k:
            break

        if pred in truths_set:
            dcg += log2_inv[i]
            truths_set.remove(pred)

    return dcg / idcg


def get_ndcg(pred: dict[int, list[int]], truth: dict[int, list[int]],
             k: int) -> float:
    """
    Compute the Normalized Discounted Cumulative Gain (NDCG) for a given prediction and truth.
    src: https://recpack.froomle.ai/generated/recpack.metrics.NDCGK.html
    :param pred: list of predicted items per user
    :param truth: list of ground truth items per user
    :param k: number of items to consider for the NDCG
    :return: NDCG@k
    """
    if k == 0:
        return 1.0

    # precompute a list of idcg
    log2_inv = [1 / math.log2(i + 2) for i in range(k)]
    idcg_ls = np.cumsum(log2_inv)

    # compute the NDCG per user
    total_ndcg = 0
    for user in pred.keys():
        if user not in truth:
            continue

        total_ndcg += get_ndcg_user(pred[user], truth[user], k, idcg_ls,
                                    log2_inv)

    return total_ndcg / len(truth.keys())


def _get_calibrated_recall_user(preds: list[int], truths: list[int],
                                k: int) -> float:
    # depending on the list that is empty, the recall is 0 or 1
    if not len(truths):
        return 1.0
    elif not len(preds):
        return 0.0

    truths_set = set(truths)
    if len(preds) < k:
        k = len(preds)

    recall = 0
    for i in range(k):
        if preds[i] in truths_set:
            recall += 1
            truths_set.remove(preds[i])

    return recall / min(len(truths), k)


def get_calibrated_recall(pred: dict[int, list[int]],
                          truth: dict[int, list[int]], k: int) -> float:
    """
    Compute the recall of a given prediction.
    src: https://recpack.froomle.ai/generated/recpack.metrics.CalibratedRecallK.html#recpack.metrics.CalibratedRecallK
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param truth: list of ground truth items per user
    :param k: number of items to consider for the recall
    :return: recall@k
    """
    # if nothing needs to be recalled the recall is 1
    if k == 0:
        return 1.0

    total_recall = 0
    for user in pred.keys():
        if user not in truth:
            continue
        total_recall += _get_calibrated_recall_user(pred[user], truth[user], k)

    # beware of dividing it by all the users
    return total_recall / len(truth.keys())


'''
########################    DIVERSITY@20    ########################
'''


def get_coverage(pred: dict[int, list[int]], n_items: int, k: int) -> float:
    """
    Compute the coverage of a given prediction.
    src: https://dl.acm.org/doi/pdf/10.1145/1060745.1060754 (3.2.1)
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param n_items: number of all possible items
    :param k: number of items to consider in the prediction
    :return: coverage@k
    """
    items = set()
    for user in pred.keys():
        for i in range(min(k, len(pred[user]))):
            items.add(pred[user][i])

    return len(items) / n_items


def _get_intra_list_similarity_user(preds: list[int], item_sim: np.ndarray,
                                    k: int) -> float:
    """
    calculate intra list similarity using the history of item interactions
    :param preds: the predictions for a single user
    :param item_sim: the similarity between items
    :param k: the number of items to consider in the predictions
    :return: intra list diversity for a user
    """
    if k == 1 or len(preds) == 1:
        return 1.0
    if len(preds) < k:
        k = len(preds)

    intra_list_similarity = 0
    for i in range(k):
        for j in range(i + 1, k):
            if preds[i] >= item_sim.shape[0] or preds[j] >= item_sim.shape[0]:
                print('out of bounds')
                continue
            intra_list_similarity += item_sim[preds[i], preds[j]]

    if k == 2 or len(preds) == 2:
        return intra_list_similarity

    denominator = (k - 1) * (k - 2) / 2
    return intra_list_similarity / denominator


def get_intra_list_similarity(pred: dict[int, list[int]], ui_mat,
                              k: int) -> float:
    """
    calculate intra list diversity using information about the items
    define an item by its user-item interaction vector
    src: https://dl.acm.org/doi/pdf/10.1145/1060745.1060754 (3.3)
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param ui_mat: user-item interaction matrix
    :param k: the number of items to compare to each other in the list
    :return: the average intra list diversity
    """
    # relevant users
    relevant_users = list(pred.keys())

    # pairwise cosine similarity btwn items
    item_dist = sklearn.metrics.pairwise.cosine_similarity(ui_mat.T)

    intra_list_similarity = 0
    for user in relevant_users:
        if user not in pred:
            continue
        intra_list_similarity += _get_intra_list_similarity_user(pred[user],
                                                                 item_dist, k)

    # average over all users
    return intra_list_similarity / len(relevant_users)


'''
########################    FAIRNESS@20    ########################
'''


def calc_gini_index(x: np.array) -> float:
    """
    src: https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python/48999797#48999797
    :param x: a numpy array of the values
    :return: gini index of array of values
    """
    if len(x) <= 1:
        return 0.0

    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)

    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def get_gini_index(pred: dict[int, list[int]], k: int) -> float:
    """
    Compute the Gini index for all predictions by getting the number of item
    occurrences in the top k predictions.
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :return: gini index @ k
    """
    cntr = Counter()
    for preds in pred.values():
        top_k_preds = preds[:k]
        cntr += Counter(top_k_preds)

    # calculate the gini index based on the counter values
    vals = list(cntr.values())

    return calc_gini_index(np.asarray(vals))


def get_publisher_fairness(pred: dict, i2p_dict: dict, k: int) -> float:
    """
    work with publisher fairness, so that each publisher has the same
    representation in the top-k recommendations
    use gini index to measure the fairness
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param k: the number of items to consider in the predictions
    :return: the gini index grouped by publishers
    """
    if k == 0:
        return 0.0

    # group the predictions per publisher
    pred_per_publisher = Counter()
    for user in pred.keys():
        preds = pred[user][:min(k, len(pred[user]))]
        if not len(preds):
            continue

        publisher_preds = map(lambda i: i2p_dict[i], preds)
        pred_per_publisher += Counter(publisher_preds)

    # calculate the gini index
    gini = calc_gini_index(list(pred_per_publisher.values()))

    return gini


'''
########################    NOVELTY@20    ########################
'''


def get_novelty(pred: dict[int, list[int]], train_ui_mat: csr_matrix,
                test_in_ui_mat: csr_matrix, k: int) -> float:
    """
    src: https://dl.acm.org/doi/pdf/10.1145/1944339.1944341 (3.2)
    we use cosine distance as the distance measure
    :param pred: the predictions in dictionary format with uid=key & iid=vals
    :param train_ui_mat: the user item interaction matrix
    :param k: the number of items to consider in the predictions
    :param test_in_ui_mat: the user item interaction matrix with test fold-in
    :return: the average novelty
    """
    if k == 0:
        return 0.0

    # pairwise cosine distance btwn items, using only training interactions
    item_dist = sklearn.metrics.pairwise.cosine_distances(train_ui_mat.T)

    total_novelty = 0.0
    for u, items in pred.items():
        # limit the items interacted with to the top-k
        top_k_items = items[:k]

        # L contains the items that a user has interacted with
        # get nonzero indices of the user-item interaction matrix for user u
        L = np.nonzero(test_in_ui_mat[u])[1]

        user_novelty = 0.0
        cnt = 0
        for pred_item in top_k_items:
            for all_item in L:
                if all_item == pred_item:
                    continue

                cnt += 1
                item_novelty = item_dist[pred_item, all_item]
                user_novelty += item_novelty

        # add the user's novelty to the total novelty
        total_novelty += user_novelty / max(cnt, 1)

    return total_novelty / len(pred.keys())  # average over all users



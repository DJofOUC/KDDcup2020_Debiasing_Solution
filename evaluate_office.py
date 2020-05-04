# coding=utf-8
import datetime
import json
import sys
import time
from collections import defaultdict

import numpy as np

def report_error(stdout, info):
    print(stdout, info)

def report_score(
        stdout, score=0.0,
        ndcg_50_full=0.0, ndcg_50_half=0.0,
        hitrate_50_full=0.0, hitrate_50_half=0.0):
    print(stdout,
          "score={}, ndcg_50_full={}, ndcg_50_half={}, hitrate_50_full={}, hitrate_50_half={}".format(
              score, ndcg_50_full, ndcg_50_half, hitrate_50_full, hitrate_50_half))

# the higher scores, the better performance
def evaluate_each_phase(predictions, answers):
    """

    :param predictions:
    :param answers: 字典类型，{"11":[1234, 0.1], "22":[1235, 0.2]},键是user_id,值是包含item_id,item_degree的列表
    :return:
    """
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id] #获取商品id和分数
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2] #商品的评分的中位数，

    num_cases_full = 0.0 #所有用户的点击
    ndcg_50_full = 0.0 #总体推荐的ndcg
    ndcg_50_half = 0.0 #总体推荐中的后一半商品（姑且认为是冷门商品）的ndcg
    num_cases_half = 0.0 #总共点击的商品次数
    hitrate_50_full = 0.0 #预测命中的商品个数
    hitrate_50_half = 0.0 #预测命中的后一半商品个数，衡量对于冷门商品的推荐
    for user_id in answers: #获取每个用户的实际点击
        item_id, item_degree = answers[user_id]
        rank = 0

        # rank=0表示第一次点击，这个条件是对该用户推荐的top50中第几个（rank）预测命中了该商品
        while rank < 50 and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0 #总共点击的商品个数
        if rank < 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0) #加上dcg
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree: #命中的商品是后一半半热门的商品
            num_cases_half += 1.0 #总共点击的后一半的商品个数
            if rank < 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)

# submit_fname is the path to the file submitted by the participants.
# debias_track_answer.csv is the standard answer, which is not released.
def evaluate(stdout, submit_fname,
             answer_fname='debias_track_answer.csv', current_time=None):
    schedule_in_unix_time = [
        0,  # ........ 1970-01-01 08:00:00 (T=0)
        1586534399,  # 2020-04-10 23:59:59 (T=1)
        1587139199,  # 2020-04-17 23:59:59 (T=2)
        1587743999,  # 2020-04-24 23:59:59 (T=3)
        1588348799,  # 2020-05-01 23:59:59 (T=4)
        1588953599,  # 2020-05-08 23:59:59 (T=5)
        1589558399,  # 2020-05-15 23:59:59 (T=6)
        1590163199,  # 2020-05-22 23:59:59 (T=7)
        1590767999,  # 2020-05-29 23:59:59 (T=8)
        1591372799  # .2020-06-05 23:59:59 (T=9)
    ]
    assert len(schedule_in_unix_time) == 10
    for i in range(1, len(schedule_in_unix_time) - 1):
        # 604800 == one week
        assert schedule_in_unix_time[i] + 604800 == schedule_in_unix_time[i + 1]

    if current_time is None:
        current_time = int(time.time())
    print('current_time:', current_time)
    print('date_time:', datetime.datetime.fromtimestamp(current_time))
    current_phase = 0
    while (current_phase < 9) and (
            current_time > schedule_in_unix_time[current_phase + 1]):
        current_phase += 1
    print('current_phase:', current_phase)

    try:
        answers = [{} for _ in range(10)]
        #获取真实的点击记录
        with open(answer_fname, 'r') as fin:
            for line in fin:
                line = [int(x) for x in line.split(',')]
                phase_id, user_id, item_id, item_degree = line
                assert user_id % 11 == phase_id
                # exactly one test case for each user_id
                answers[phase_id][user_id] = (item_id, item_degree)
    except Exception as _:
        return report_error(stdout, 'server-side error: answer file incorrect')

    try:
        predictions = {}
        #读取提交的预测文件
        with open(submit_fname, 'r') as fin:
            for line in fin:
                line = line.strip()
                if line == '':
                    continue
                line = line.split(',')
                user_id = int(line[0])
                if user_id in predictions:
                    return report_error(stdout, 'submitted duplicate user_ids')
                item_ids = [int(i) for i in line[1:]]
                if len(item_ids) != 50:
                    return report_error(stdout, 'each row need have 50 items')
                if len(set(item_ids)) != 50:
                    return report_error(
                        stdout, 'each row need have 50 DISTINCT items')
                predictions[user_id] = item_ids
    except Exception as _:
        return report_error(stdout, 'submission not in correct format')

    scores = np.zeros(4, dtype=np.float32)

    # The final winning teams will be decided based on phase T=7,8,9 only.
    # We thus fix the scores to 1.0 for phase 0,1,2,...,6 at the final stage.
    if current_phase >= 7:  # if at the final stage, i.e., T=7,8,9
        scores += 7.0  # then fix the scores to 1.0 for phase 0,1,2,...,6
    phase_beg = (7 if (current_phase >= 7) else 0)
    phase_end = current_phase + 1
    #for phase_id in range(phase_beg, phase_end):
    for phase_id in range(1):
        for user_id in answers[phase_id]:
            if user_id not in predictions:
                return report_error(
                    stdout, 'user_id %d of phase %d not in submission' % (
                        user_id, phase_id))
        try:
            # We sum the scores from all the phases, instead of averaging them.
            scores += evaluate_each_phase(predictions, answers[phase_id])
        except Exception as _:
            return report_error(stdout, 'error occurred during evaluation')

    return report_score(
        stdout, score=float(scores[0]),
        ndcg_50_full=float(scores[0]), ndcg_50_half=float(scores[1]),
        hitrate_50_full=float(scores[2]), hitrate_50_half=float(scores[3]))

# FYI. You can create a fake answer file for validation based on this. For example,
# you can mask the latest ONE click made by each user in underexpose_test_click-T.csv,
# and use those masked clicks to create your own validation set, i.e.,
# a fake underexpose_test_qtime_with_answer-T.csv for validation.


def _create_answer_file_for_evaluation(answer_fname='debias_track_answer.csv'):
    """
    create a file with phase_id, user_id, item_id, item_deg[item_id]
    :param answer_fname:
    :return:
    """
    train = '../raw_data/underexpose_train/underexpose_train_click-%d.csv'
    test = '../raw_data/underexpose_test/underexpose_test_click-%d.csv'

    # underexpose_test_qtime-T.csv contains only <user_id, item_id>
    # underexpose_test_qtime_with_answer-T.csv contains <user_id, item_id, time>
    answer = '../raw_data/underexpose_test_with_answer/underexpose_test_qtime_with_answer-%d.csv'  # not released

    item_deg = defaultdict(lambda: 0)
    with open(answer_fname, 'w') as fout:
        for phase_id in range(1):
            with open(train % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1 #训练集商品总点击次数，degress 分数
            with open(test % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    item_deg[item_id] += 1 #再加上 测试集商品点击次数
            with open(answer % phase_id) as fin:
                for line in fin:
                    user_id, item_id, timestamp = line.split(',')
                    user_id, item_id, timestamp = (
                        int(user_id), int(item_id), float(timestamp))
                    assert user_id % 11 == phase_id #与user id的制定有关
                    print(phase_id, user_id, item_id, item_deg[item_id],
                          sep=',', file=fout)

if __name__ == "__main__":
    #_create_answer_file_for_evaluation()
    evaluate("stdout",submit_fname="../submission/underexpose_submit-0.csv",)
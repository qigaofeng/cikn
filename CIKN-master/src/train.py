import tensorflow as tf
import numpy as np
from model import CIKN


def train(args, data, show_loss, show_topk):
    # print(data)
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]
    ripple_set = data[9]
    # print(ripple_set)
    interaction_table, offset = get_interaction_table(train_data, n_entity)
    model = CIKN(args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset)

    # add
    # saver = tf.train.Saver(max_to_keep=10)
    # model_file = tf.train.latest_checkpoint('train_model/' + args.dataset)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        interaction_table.init.run()
        for step in range(args.n_epochs):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(args, model, train_data, ripple_set, start,
                                                          start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))

            # CTR evaluation
            train_auc, train_f1, train_acc = ctr_eval(sess, args, model, train_data, ripple_set, args.batch_size)
            eval_auc, eval_f1, eval_acc = ctr_eval(sess, args, model, eval_data, ripple_set, args.batch_size)
            test_auc, test_f1, test_acc = ctr_eval(sess, args, model, test_data, ripple_set, args.batch_size)

            print(
                'epoch %d    train auc: %.4f  f1: %.4f   acc: %.4f    eval auc: %.4f  f1: %.4f  acc: %.4f    test auc: %.4f  f1: %.4f  acc: %.4f'
                % (step, train_auc, train_f1, train_acc, eval_auc, eval_f1, eval_acc, test_auc, test_f1, test_acc))

            # # find bad case
            # train_err_pred = ctr_survey(sess, model, train_data, args.batch_size)
            # eval_err_pred = ctr_survey(sess, model, eval_data, args.batch_size)
            # test_err_pred = ctr_survey(sess, model, test_data, args.batch_size)

            # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, ripple_set, args.batch_size,
                    start)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')

            # 每一轮都保存模型参数
            # save_path = saver.save(sess, "./ckpt_model/keypoint_model.ckpt", global_step=step)
            # print("model has saved,saved in path: %s" % save_path)
            #
            # # 加载数据
            # model_file = tf.train.latest_checkpoint('ckpt_model/')
            # saver.restore(sess, model_file)

            # print(train_err_pred)

            # with open('observation/train_err_pred.txt', 'w') as file:
            #     for i in range(len(train_err_pred)):
            #         s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
            #         file.write(s)
            # with open('observation/eval_err_pred.txt', 'w') as file:
            #     for i in range(len(eval_err_pred)):
            #         s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
            #         file.write(s)
            # with open('observation/test_err_pred.txt', 'w') as file:
            #     for i in range(len(test_err_pred)):
            #         s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
            #         file.write(s)
            #
            # print('finish.')
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #
    #     model_file = tf.train.latest_checkpoint('ckpt_model/')
    #     saver.restore(sess, model_file)
    #
    #     # print(train_err_pred)
    #
    #     with open('observation/train_err_pred.txt', 'w') as file:
    #         for i in range(len(train_err_pred)):
    #             s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
    #             file.write(s)
    #     with open('observation/eval_err_pred.txt', 'w') as file:
    #         for i in range(len(eval_err_pred)):
    #             s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
    #             file.write(s)
    #     with open('observation/test_err_pred.txt', 'w') as file:
    #         for i in range(len(test_err_pred)):
    #             s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
    #             file.write(s)
    #
    #     print('finish.')


def get_interaction_table(train_data, n_entity):
    offset = len(str(n_entity))
    offset = 10 ** offset
    keys = train_data[:, 0] * offset + train_data[:, 1]
    keys = keys.astype(np.int64)
    values = train_data[:, 2].astype(np.float32)

    interaction_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
    return interaction_table, offset


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_indices: data[start:end, 1],
                 model.labels: data[start:end, 2]}
    for i in range(args.n_iter):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def ctr_eval(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    f1_list = []
    acc_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1, acc = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        acc_list.append(acc)
        start += batch_size
    return float(np.mean(auc_list)), float(np.mean(f1_list)), float(np.mean(acc_list))


def topk_eval(args, data, sess, model, user_list, train_record, test_record, item_set, k_list, ripple_set, start):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for i in range(args.n_iter):
        model.memories_h[i] = [ripple_set[user][i][0] for user in data[start:start + args.batch_size]]
        model.memories_r[i] = [ripple_set[user][i][1] for user in data[start:start + args.batch_size]]
        model.memories_t[i] = [ripple_set[user][i][2] for user in data[start:start + args.batch_size]]

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + args.batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * args.batch_size,
                                                    model.item_indices: test_item_list[start:start + args.batch_size],
                                                    model.memories_h: model.memories_h,
                                                    model.memories_r: model.memories_r,
                                                    model.memories_t: model.memories_t})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += args.batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * args.batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               args.batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

# def ctr_survey(sess, model, data, batch_size):
#     start = 0
#     error_index_array = None
#     while start + batch_size <= data.shape[0]:
#         scores, labels = model.survey(sess, get_feed_dict(model, data, start, start + batch_size))
#
#         err = np.nonzero(scores - labels)  # 这个batch中预测错了的index
#         if start != 0:
#             error_index_array = np.concatenate([error_index_array, [x + start for x in err]], axis=1)
#         else:
#             error_index_array = err
#
#         start += batch_size

# print(len(data))  # 8469
# print(error_index_array)
# print(error_index_array.shape)  # (1, 2318)
# return [(data[index][0], data[index][1]) for index in error_index_array[0]]

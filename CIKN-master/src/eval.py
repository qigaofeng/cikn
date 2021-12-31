# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#

# import tensorflow as tf
# import numpy as np
# from model import CIKN
#
#
# def get_feed_dict(model, data, start, end):
#     feed_dict = {model.user_indices: data[start:end, 0],
#                  model.item_indices: data[start:end, 1],
#                  model.labels: data[start:end, 2]}
#     return feed_dict
#
#
# def ctr_eval(sess, model, data, batch_size):
#     start = 0
#     auc_list = []
#     f1_list = []
#     while start + batch_size <= data.shape[0]:
#         auc, f1 = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
#         auc_list.append(auc)
#         f1_list.append(f1)
#         start += batch_size
#     return float(np.mean(auc_list)), float(np.mean(f1_list))
#
#
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
#
#     # print(len(data))  # 8469
#     # print(error_index_array)
#     # print(error_index_array.shape)  # (1, 2318)
#     return [(data[index][0], data[index][1]) for index in error_index_array[0]]
#
#
# def survey(args, data):
#     n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
#     train_data, eval_data, test_data = data[4], data[5], data[6]
#     adj_entity, adj_relation = data[7], data[8]
#     interaction_table, offset = get_interaction_table(train_data, n_entity)
#     print("n_user", n_user, "n_item", n_item, "n_entity", n_entity, "n_relation", n_relation)
#
#     model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation,interaction_table, offset)
#
#     saver = tf.train.Saver(max_to_keep=1)
#     model_file = tf.train.latest_checkpoint('train_model/'+args.dataset)
#
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         saver.restore(sess, model_file)
#
#         # train_auc, train_f1 = ctr_eval(sess, model, train_data, args.batch_size)
#         # eval_auc, eval_f1 = ctr_eval(sess, model, eval_data, args.batch_size)
#         # test_auc, test_f1 = ctr_eval(sess, model, test_data, args.batch_size)
#
#         # print('train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
#         #       % (train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))
#
#         train_err_pred = ctr_survey(sess, model, train_data, args.batch_size)
#         eval_err_pred = ctr_survey(sess, model, eval_data, args.batch_size)
#         test_err_pred = ctr_survey(sess, model, test_data, args.batch_size)
#         # print(train_err_pred)
#
#         with open('observation/train_err_pred.txt', 'w') as file:
#             for i in range(len(train_err_pred)):
#                 s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
#                 file.write(s)
#         with open('observation/eval_err_pred.txt', 'w') as file:
#             for i in range(len(eval_err_pred)):
#                 s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
#                 file.write(s)
#         with open('observation/test_err_pred.txt', 'w') as file:
#             for i in range(len(test_err_pred)):
#                 s = str(train_err_pred[i]).replace('(', '').replace(')', '') + '\n'
#                 file.write(s)
#
#         print('finish.')
#
#
# def get_interaction_table(train_data, n_entity):
#     offset = len(str(n_entity))
#     offset = 10 ** offset
#     keys = train_data[:, 0] * offset + train_data[:, 1]
#     keys = keys.astype(np.int64)
#     values = train_data[:, 2].astype(np.float32)
#
#     interaction_table = tf.contrib.lookup.HashTable(
#         tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=values), default_value=0.5)
#     return interaction_table, offset
#
#

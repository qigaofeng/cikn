import numpy as np
import tensorflow as tf
from tensorflow.python import Zeros
from tensorflow.python.keras.initializers import glorot_normal

from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator, LabelAggregator
from sklearn.metrics import f1_score, roc_auc_score
from tensorflow.python.keras.layers import (Dense, Embedding, Lambda, add,
                                            multiply, Layer)
from tensorflow.python.keras.regularizers import l2


class ATT(Layer):
    """
      Input shape
        - A list of 3D tensor with shape: ``(batch_size,1,embedding_size)``.

      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.

    """

    def __init__(self, attention_factor=4, l2_reg_w=0, keep_prob=0.5, seed=1024, **kwargs):
        self.attention_factor = attention_factor
        self.l2_reg_w = l2_reg_w
        self.keep_prob = keep_prob
        self.seed = seed
        super(ATT, self).__init__(**kwargs)

    def build(self, input_shape):
        # if not isinstance(input_shape, list) or len(input_shape) < 2:
        #    raise ValueError('A `AttentionalFM` layer should be called '
        #                     'on a list of at least 2 inputs')

        #         shape_set = set()
        #         reduced_input_shape = [shape.as_list() for shape in input_shape]
        #         for i in range(len(input_shape)):
        #             shape_set.add(tuple(reduced_input_shape[i]))

        #         if len(shape_set) > 1:
        #             raise ValueError('A `AttentionalFM` layer requires '
        #                              'inputs with same shapes '
        #                              'Got different shapes: %s' % (shape_set))

        #         if len(input_shape[0]) != 3 or input_shape[0][1] != 1:
        #             raise ValueError('A `AttentionalFM` layer requires '
        #                              'inputs of a list with same shape tensor like\
        #                              (None, 1, embedding_size)'
        #                              'Got different shapes: %s' % (input_shape[0]))

        embedding_size = input_shape[-1].value

        self.attention_W = self.add_weight(shape=(embedding_size,
                                                  self.attention_factor), initializer=glorot_normal(seed=self.seed),
                                           regularizer=l2(self.l2_reg_w), name="attention_W")
        self.attention_b = self.add_weight(
            shape=(self.attention_factor,), initializer=Zeros(), name="attention_b")
        self.projection_h = self.add_weight(shape=(self.attention_factor, 1),
                                            initializer=glorot_normal(seed=self.seed), name="projection_h")

        # Be sure to call this somewhere!
        super(ATT, self).build(input_shape)

    def call(self, inputs, **kwargs):
        inner_product = inputs  # concat_fun(ans,axis=1)

        bi_interaction = inner_product

        attention_temp = tf.nn.relu(tf.nn.bias_add(tf.tensordot(
            bi_interaction, self.attention_W, axes=(-1, 0)), self.attention_b))
        #  Dense(self.attention_factor,'relu',kernel_regularizer=l2(self.l2_reg_w))(bi_interaction)
        self.normalized_att_score = tf.nn.softmax(tf.tensordot(
            attention_temp, self.projection_h, axes=(-1, 0)), dim=1)
        attention_output = self.normalized_att_score * bi_interaction
        return attention_output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `AFMLayer` layer should be called '
                             'on a list of inputs.')
        return (None, self.attention_factor)

    def get_config(self, ):
        config = {'attention_factor': self.attention_factor,
                  'l2_reg_w': self.l2_reg_w, 'keep_prob': self.keep_prob, 'seed': self.seed}
        base_config = super(ATT, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CIKN(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation, interaction_table, offset):

        self._parse_args(args, adj_entity, adj_relation, interaction_table, offset)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, interaction_table, offset):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.interaction_table = interaction_table
        self.offset = offset
        self.ls_weight = args.ls_weight
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.kge_weight = args.kge_weight

        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')

        # add user_三元组
        self.memories_h = []  # add
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_iter):
            self.memories_h.append(
                tf.placeholder(dtype=tf.int64, shape=[None, self.n_neighbor], name="memories_h_" + str(hop)))
            self.memories_r.append(
                tf.placeholder(dtype=tf.int64, shape=[None, self.n_neighbor], name="memories_r_" + str(hop)))
            self.memories_t.append(
                tf.placeholder(dtype=tf.int64, shape=[None, self.n_neighbor], name="memories_t_" + str(hop)))

    def _build_model(self, n_user, n_entity, n_relation):
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(name="transform_matrix", shape=[self.dim, self.dim], dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer())

        #  Embedding
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=CIKN.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=CIKN.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation, self.dim], initializer=CIKN.get_initializer(), name='relation_emb_matrix')

        self.relation_emb_matrix_user = tf.get_variable(
            shape=[n_relation, self.dim, self.dim], initializer=CIKN.get_initializer(),
            name="relation_emb_matrix_user", )

        # [batch size, dim]
        # old item_embedding
        self.item_embedding = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)

        # old user_embedding
        self.user_embedding = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}

        # Calculate item_embeddings
        entities, relations = self.get_neighbors(self.item_indices)

        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)
        # self.item_embedding = ATT()(self.item_embedding)

        # add_regularization
        self._regularization(entities, relations)

        # Calculate user_embeddings
        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_iter):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i]))

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(tf.nn.embedding_lookup(self.relation_emb_matrix_user, self.memories_r[i]))

            # [batch size, n_memory, dim]
            self.t_emb_list.append(tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i]))

        # new user_embeddings
        user_embeddings = self._key_addressing()  # user_embeddings



        # o_list = ATT()(o_list)

        # [batch_size]
        self.scores = tf.squeeze(self.predict(self.item_embeddings, user_embeddings))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        cf_list = []
        for hop in range(self.n_iter):
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embedding, axis=2)
            u = tf.expand_dims(self.user_embedding, axis=2)

            # [batch_size, n_memory]
            probs_v = tf.squeeze(tf.matmul(Rh, v), axis=2)
            probs_u = tf.squeeze(tf.matmul(Rh, u), axis=2)
            # [batch_size, n_memory]
            probs_normalized_v = tf.nn.softmax(probs_v)
            probs_normalized_u = tf.nn.softmax(probs_u)
            # print(probs_normalized)
            # if probs_normalized == 0:
            #     mask_2.append(probs_normalized)
            # [batch_size, n_memory, 1]
            probs_expanded_v = tf.expand_dims(probs_normalized_v, axis=2)
            probs_expanded_u = tf.expand_dims(probs_normalized_u, axis=2)
            # [batch_size, dim]
            # o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded_v, axis=1)
            o_cf = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded_v * probs_expanded_u, axis=1)
            self.item_embeddings = tf.matmul(self.item_embeddings + o_cf, self.transform_matrix)

            cf_list.append(o_cf)
        # o_list_u = tf.concat([o_list, u_list], axis=0)
        # o_list_u = tf.reshape(tf.concat([o_list, u_list], axis=-1), [-1, self.dim])
        # print(o_list_u) # Tensor("concat:0", shape=(1, ?, 8), axis=-1 ; axis=1:shape=(1, ?, 4); axis=0:shape=(2, ?, 4))
        # print(o_list)  # [<tf.Tensor 'Sum:0' shape=(?, 4) dtype=float32>]
        # print("mask_2", mask_2)
        return cf_list

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]  # item_embeddings
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = SumAggregator(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = SumAggregator(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embedding, masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _regularization(self, entities, relations):

        # calculate initial labels; calculate updating masks for label propagation
        entity_labels = []
        reset_masks = []  # True means the label of this item is reset to initial value during label propagation
        holdout_item_for_user = None

        for entities_per_iter in entities:
            # [batch_size, 1]
            users = tf.expand_dims(self.user_indices, 1)
            # [batch_size, n_neighbor^i]
            user_entity_concat = users * self.offset + entities_per_iter

            # the first one in entities is the items to be held out
            if holdout_item_for_user is None:
                holdout_item_for_user = user_entity_concat

            # [batch_size, n_neighbor^i]
            initial_label = self.interaction_table.lookup(user_entity_concat)
            holdout_mask = tf.cast(holdout_item_for_user - user_entity_concat, tf.bool)  # False if the item is held out
            reset_mask = tf.cast(initial_label - tf.constant(0.5), tf.bool)  # True if the entity is a labeled item
            reset_mask = tf.logical_and(reset_mask, holdout_mask)  # remove held-out items
            initial_label = tf.cast(holdout_mask, tf.float32) * initial_label + tf.cast(
                tf.logical_not(holdout_mask), tf.float32) * tf.constant(0.5)  # label initialization

            reset_masks.append(reset_mask)
            entity_labels.append(initial_label)
        reset_masks = reset_masks[:-1]  # we do not need the reset_mask for the last iteration

        # label propagation
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        aggregator = LabelAggregator(self.batch_size, self.dim)
        for i in range(self.n_iter):
            entity_labels_next_iter = []
            for hop in range(self.n_iter - i):
                vector = aggregator(self_vectors=entity_labels[hop],
                                    neighbor_vectors=tf.reshape(
                                        entity_labels[hop + 1], [self.batch_size, -1, self.n_neighbor]),
                                    neighbor_relations=tf.reshape(
                                        relation_vectors[hop], [self.batch_size, -1, self.n_neighbor, self.dim]),
                                    user_embeddings=self.user_embedding,
                                    masks=reset_masks[hop])
                entity_labels_next_iter.append(vector)
            entity_labels = entity_labels_next_iter

        self.predicted_labels = tf.squeeze(entity_labels[0], axis=-1)

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.scores))

        self.kge_loss = 0
        for hop in range(self.n_iter):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded))
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        # L2 loss
        self.l2_loss = 0
        for hop in range(self.n_iter):
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop]))
            self.l2_loss += tf.reduce_mean(tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop]))

        for aggregator in self.aggregators:
            self.l2_loss += tf.nn.l2_loss(aggregator.weights)
        self.l2_loss = self.l2_weight * self.l2_loss

        # LS loss
        self.ls_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.predicted_labels))
        self.ls_loss = self.ls_weight * self.ls_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss + self.ls_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        acc = np.mean(np.equal(scores, labels))
        return auc, f1, acc

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)

    # def survey(self, sess, feed_dict):
    #     labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
    #     # auc = roc_auc_score(y_true=labels, y_score=scores)
    #     scores[scores >= 0.5] = 1
    #     scores[scores < 0.5] = 0
    #     # f1 = f1_score(y_true=labels, y_pred=scores)
    #     return scores, labels

    def predict(self, item_embeddings, cf_list):
        y = cf_list[-1]  # Tensor("strided_slice:0", shape=(?, 4), dtype=float32)
        # print("y: ", y)
        for i in range(self.n_iter - 1):
            y += cf_list[i]

        # [batch_size]
        interaction = item_embeddings * y
        final_logit = ATT()(interaction)

        # print("item_embedd : ",item_embedding)  # Tensor("att/mul:0", shape=(4096, 4), dtype=float32)
        scores = tf.reduce_sum(final_logit, axis=1)
        return scores

def activation_fun(activation, fc):

    if isinstance(activation, str):
        fc = tf.keras.layers.Activation(activation)(fc)
    elif issubclass(activation, Layer):
        fc = activation()(fc)
    else:
        raise ValueError(
            "Invalid activation,found %s.You should use a str or a Activation Layer Class." % (activation))
    return fc

class MLP(Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_size**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **keep_prob**: float between 0 and 1. Fraction of the units to keep.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self,  hidden_size, activation='relu', l2_reg=0, keep_prob=1, use_bn=False, seed=1024, **kwargs):
        self.hidden_size = hidden_size
        self.activation = activation
        self.keep_prob = keep_prob
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        super(MLP, self).__init__(**kwargs)

    def build(self, input_shape):
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_size)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i+1]),
                                        initializer=glorot_normal(
                                            seed=self.seed),
                                        regularizer=l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_size))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_size[i],),
                                     initializer=Zeros(),
                                     trainable=True) for i in range(len(self.hidden_size))]

        super(MLP, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_size)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            # fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = tf.keras.layers.BatchNormalization()(fc)
            fc = activation_fun(self.activation, fc)
            #fc = tf.nn.dropout(fc, self.keep_prob)
            fc = tf.keras.layers.Dropout(1 - self.keep_prob)(fc,)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_size) > 0:
            shape = input_shape[:-1] + (self.hidden_size[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self,):
        config = {'activation': self.activation, 'hidden_size': self.hidden_size,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'keep_prob': self.keep_prob, 'seed': self.seed}
        base_config = super(MLP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class PredictionLayer(Layer):
    """
      Arguments
         - **activation**: Activation function to use.

         - **use_bias**: bool.Whether add bias term or not.
    """

    def __init__(self, activation='sigmoid', use_bias=True, **kwargs):
        self.activation = activation
        self.use_bias = use_bias
        super(PredictionLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.use_bias:
            self.global_bias = self.add_weight(
                shape=(1,), initializer=Zeros(), name="global_bias")

        # Be sure to call this somewhere!
        super(PredictionLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_bias:
            x = tf.nn.bias_add(x, self.global_bias, data_format='NHWC')
        output = activation_fun(self.activation, x)
        output = tf.reshape(output, (-1, 1))

        return output

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self,):
        config = {'activation': self.activation, 'use_bias': self.use_bias}
        base_config = super(PredictionLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
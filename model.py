from modules import *
import tensorflow as tf
import numpy as np
# tf.enable_eager_execution() # for tensor.numpy()

def net_save(data, output_file):
    file = open(output_file, 'a')
    for i in data:
        s = str(i) + '\n'
        file.write(s)
    file.close()

class Model():
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        self.input_seq_revs = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos_revs = tf.placeholder(tf.int32, shape=(None, args.maxlen))

        self.input_seq_origin = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos_origin = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg_origin = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        
        pos = self.pos #[B, L]
        neg = self.neg #[B, L]

        pos_revs = self.pos_revs

        pos_origin = self.pos_origin
        neg_origin = self.neg_origin
        
        # aug_seq and revs_seq
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1) #[B, L, 1]
        src_masks = tf.math.equal(self.input_seq, 0)

        # org_seq
        mask_origin = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq_origin, 0)), -1) #[B, L, 1]
        src_masks_origin = tf.math.equal(self.input_seq_origin, 0)

        # label mask
        label_mask = self.label_pos_mask(args.maxlen, args.clip_k) # [1, L]

        with tf.variable_scope("SASRec", reuse=reuse):
            # sequence embedding, item embedding table
            #[B, L] -> [B, L, D]
            self.seq, item_emb_table = embedding(self.input_seq,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=tf.AUTO_REUSE)
            self.item_emb_table = item_emb_table
                                                 
            self.seq_revs, _ = embedding(self.input_seq_revs,
                                         vocab_size=itemnum + 1,
                                         num_units=args.hidden_units,
                                         zero_pad=True,
                                         scale=True,
                                         l2_reg=args.l2_emb,
                                         scope="input_embeddings",
                                         with_t=True,
                                         reuse=tf.AUTO_REUSE)

            self.seq_origin, _ = embedding(self.input_seq_origin,
                                         vocab_size=itemnum + 1,
                                         num_units=args.hidden_units,
                                         zero_pad=True,
                                         scale=True,
                                         l2_reg=args.l2_emb,
                                         scope="input_embeddings",
                                         with_t=True,
                                         reuse=tf.AUTO_REUSE)

            # Positional Encoding
            t, pos_emb_table = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                                         vocab_size=args.maxlen,
                                         num_units=args.hidden_units,
                                         zero_pad=False,
                                         scale=False,
                                         l2_reg=args.l2_emb,
                                         scope="dec_pos",
                                         reuse=tf.AUTO_REUSE,
                                         with_t=True)

            t_revs, _ = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq_revs)[1]), 0), [tf.shape(self.input_seq_revs)[0], 1]),
                                         vocab_size=args.maxlen,
                                         num_units=args.hidden_units,
                                         zero_pad=False,
                                         scale=False,
                                         l2_reg=args.l2_emb,
                                         scope="dec_pos",
                                         reuse=tf.AUTO_REUSE,
                                         with_t=True)

            t_origin, _ = embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq_origin)[1]), 0), [tf.shape(self.input_seq_origin)[0], 1]),
                                         vocab_size=args.maxlen,
                                         num_units=args.hidden_units,
                                         zero_pad=False,
                                         scale=False,
                                         l2_reg=args.l2_emb,
                                         scope="dec_pos",
                                         reuse=tf.AUTO_REUSE,
                                         with_t=True)

            self.seq += t
            self.seq_revs += t_revs
            self.seq_origin += t_origin # [B, L, D]

            # debug
            # self.debug = tf.equal(self.seq, self.seq_revs) # tf.nn.embedding_lookup(item_emb_table, self.input_seq_revs))
            # Dropout and Mask
            with tf.variable_scope('my_dropout_layer'):
                self.seq = tf.layers.dropout(self.seq,
                                             rate=args.dropout_rate,
                                             training=tf.convert_to_tensor(self.is_training), name='my_dropout')
            
            with tf.variable_scope('my_dropout_layer', reuse=True):
                if args.reversed == 1:
                    self.seq_revs = tf.layers.dropout(self.seq_revs,
                                                    rate=args.dropout_rate,
                                                    training=tf.convert_to_tensor(self.is_training), name='my_dropout') 

            with tf.variable_scope('my_dropout_layer', reuse=True):
                if args.reversed_pretrain == 1:
                    self.seq_origin = tf.layers.dropout(self.seq_origin,
                                                        rate=args.dropout_rate,
                                                        training=tf.convert_to_tensor(self.is_training), name='my_dropout') 
            
            self.seq *= mask # [B, L, D]
            self.seq_revs *= mask # [B, L, D]
            self.seq_origin *= mask_origin # [B, L, D]
            
            ##
            self.debug = tf.equal(self.seq, self.seq_revs)
            
            ## Feed to Transformer
            # Build blocks
            for i in range(args.num_blocks):
                with tf.variable_scope("num_blocks_%d" % i):
                    # Self-attention
                    self.seq = multihead_attention(queries=normalize(self.seq),
                                                   keys=self.seq,
                                                   values=self.seq,
                                                   key_masks=src_masks,
                                                   num_heads=args.num_heads,
                                                   dropout_rate=args.dropout_rate,
                                                   training=self.is_training,
                                                   causality=True,
                                                   scope="self_attention")

                    self.seq_revs = multihead_attention(queries=normalize(self.seq_revs),
                                                        keys=self.seq_revs,
                                                        values=self.seq_revs,
                                                        key_masks=src_masks,
                                                        num_heads=args.num_heads,
                                                        dropout_rate=args.dropout_rate,
                                                        training=self.is_training,
                                                        causality=True,
                                                        scope="self_attention")

                    self.seq_origin = multihead_attention(queries=normalize(self.seq_origin),
                                                          keys=self.seq_origin,
                                                          values=self.seq_origin,
                                                          key_masks=src_masks_origin,
                                                          num_heads=args.num_heads,
                                                          dropout_rate=args.dropout_rate,
                                                          training=self.is_training,
                                                          causality=True,
                                                          scope="self_attention")
                    
                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units])
                    self.seq *= mask
                    # self.debug = tf.equal(mask, mask_origin)

                    # Reverse Feed forward
                    self.seq_revs = feedforward(normalize(self.seq_revs), num_units=[args.hidden_units, args.hidden_units])
                    self.seq_revs *= mask
                   
                    # Original Feed forward
                    self.seq_origin = feedforward(normalize(self.seq_origin), num_units=[args.hidden_units, args.hidden_units])
                    self.seq_origin *= mask_origin
                    
            self.seq = normalize(self.seq) #[B, L, D]
            self.seq_revs = normalize(self.seq_revs) #[B, L, D]
            self.seq_origin = normalize(self.seq_origin) #[B, L, D]
            
        # label mask
        if args.reversed_pretrain == 1:
            pos *= label_mask
            neg *= label_mask
            pos_revs *= label_mask
            pos_origin *= label_mask
            neg_origin *= label_mask

        # reshape the matrix
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen]) #[B*L]
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen]) #[B*L]
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos) #[B*L, D]
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg) #[B*L, D]
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units]) #[B*L, D]

        pos_revs = tf.reshape(pos_revs, [tf.shape(self.input_seq_revs)[0] * args.maxlen]) #[B*L]
        pos_emb_revs = tf.nn.embedding_lookup(item_emb_table, pos_revs) #[B*L, D]
        seq_emb_revs = tf.reshape(self.seq_revs, [tf.shape(self.input_seq_revs)[0] * args.maxlen, args.hidden_units]) #[B*L, D]
        
        pos_origin = tf.reshape(pos_origin, [tf.shape(self.input_seq_origin)[0] * args.maxlen]) #[B*L]
        neg_origin = tf.reshape(neg_origin, [tf.shape(self.input_seq_origin)[0] * args.maxlen]) #[B*L]
        pos_emb_origin = tf.nn.embedding_lookup(item_emb_table, pos_origin) # [B*L, D]
        neg_emb_origin = tf.nn.embedding_lookup(item_emb_table, neg_origin) # [B*L, D]
        seq_emb_origin = tf.reshape(self.seq_origin, [tf.shape(self.input_seq_origin)[0] * args.maxlen, args.hidden_units]) #[B*L, D]

        # calculate the predict probability of all items
        test_item_emb = item_emb_table #[item_num, D]
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb)) #[B*L, item_num]
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum+1]) ##[B, L, item_num]
        self.test_logits = self.test_logits[:, -1, :] #[B, item_num]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1) #[B*L, D] -> [B*L] element-wise multiply to obtain the similarity of two vectors
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
                    
        self.pos_logits_revs = tf.reduce_sum(pos_emb_revs * seq_emb_revs, -1) #[B*L, D] -> [B*L] element-wise multiply to obtain the similarity of two vectors

        self.pos_logits_origin = tf.reduce_sum(pos_emb_origin * seq_emb_origin, -1) #[B*L, D] -> [B*L] element-wise multiply to obtain the similarity of two vectors
        self.neg_logits_origin = tf.reduce_sum(neg_emb_origin * seq_emb_origin, -1)

        # ignore padding items (0) and calculate loss
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen]) #[B*L]
        self.loss_left = tf.reduce_sum(- tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget - tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss_aug = self.loss_left + sum(reg_losses) 
        
        tf.summary.scalar('loss_left', self.loss_left)
        tf.summary.scalar('aug loss', self.loss_aug) # loss_left + loss_norm
        
        # @BiCAT Fine-tuning
        if args.reversed_pretrain == 1:
            # original sequences
            # istarget_origin = tf.reshape(tf.to_float(tf.not_equal(pos_origin, 0)), [tf.shape(self.input_seq_origin)[0] * args.maxlen]) #[B*L]
            # self.loss_origin = tf.reduce_sum(- tf.log(tf.sigmoid(self.pos_logits_origin) + 1e-24) * istarget_origin - tf.log(1 - tf.sigmoid(self.neg_logits_origin) + 1e-24) * istarget_origin
            # ) / tf.reduce_sum(istarget_origin)
            
            mat_shape = [tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units]
            self.kl_loss = self.compute_kl_loss(seq_emb, seq_emb_origin, mat_shape)
            self.loss = self.loss_aug + args.alpha_coef * self.kl_loss # + self.loss_origin # hyper-parameter
            
            # tf.summary.scalar('origin loss', self.loss_origin)
            tf.summary.scalar('kl_loss', self.kl_loss)
            
        # @Check debug network
        # variables_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # net_save(variables_net, './network_param.txt')

        # @BiCAT Pre-training
        if args.reversed == 1:
            self.loss_right = tf.reduce_sum(- tf.log(tf.sigmoid(self.pos_logits_revs) + 1e-24) * istarget - tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)
            tf.summary.scalar('loss_right', self.loss_right)
            self.loss = self.loss_aug + args.lambda_coef * self.loss_right # hyper-parameter


        self.auc = tf.reduce_sum(((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def compute_kl_loss(self, aug_tea, org_stu, mat_shape):
        # aug_tea = tf.matmul(aug_tea, tf.transpose(item_emb_table))  # [B*L, item_num]
        # org_stu = tf.matmul(org_stu, tf.transpose(item_emb_table))
        
        aug_tea = tf.nn.softmax(aug_tea, axis=-1) # [B*L, item_num]
        org_stu = tf.nn.softmax(org_stu, axis=-1)
        
        aug_tea_loss = tf.reduce_sum(aug_tea * tf.math.log(aug_tea / org_stu), axis=1)
        # zero_mat = tf.zeros_like(aug_tea_loss)
        # aug_tea_loss = tf.where(tf.math.is_nan(aug_tea_loss), zero_mat, aug_tea_loss)

        org_stu_loss = tf.reduce_sum(org_stu * tf.math.log(org_stu / aug_tea), axis=1)
        # org_stu_loss = tf.where(tf.math.is_nan(org_stu_loss), zero_mat, org_stu_loss)

        aug_tea_loss = tf.reduce_sum(aug_tea_loss)
        org_stu_loss = tf.reduce_sum(org_stu_loss)
        # print(p_tea_loss, p_stu_loss)
        # input('check')

        kl_loss = (aug_tea_loss + org_stu_loss) / 2
        return kl_loss

    def label_pos_mask(self, maxlen, last_k):
        label_mask = np.zeros([1, maxlen]) # [1, L]
        label_mask[:, -last_k:] = 1
        label_mask = tf.convert_to_tensor(label_mask, dtype=tf.int32)
        return label_mask
        
    def predict(self, sess, u, seq):
        return sess.run(self.test_logits, {self.u: u, self.input_seq: seq, self.is_training: False})

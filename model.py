from modules import *
import tensorflow as tf

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
        
        pos = self.pos #[B, L]
        neg = self.neg #[B, L]
        pos_revs = self.pos_revs
        
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1) #[B, L, 1]
        src_masks = tf.math.equal(self.input_seq, 0)

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
                                                 reuse=tf.AUTO_REUSE
                                                 )
            
            self.seq_revs, item_emb_table_reverse = embedding(self.input_seq_revs,
                                                 vocab_size=itemnum + 1,
                                                 num_units=args.hidden_units,
                                                 zero_pad=True,
                                                 scale=True,
                                                 l2_reg=args.l2_emb,
                                                 scope="input_embeddings",
                                                 with_t=True,
                                                 reuse=tf.AUTO_REUSE
                                                 )

            # Positional Encoding
            t, pos_emb_table = embedding(
                                         tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
                                         vocab_size=args.maxlen,
                                         num_units=args.hidden_units,
                                         zero_pad=False,
                                         scale=False,
                                         l2_reg=args.l2_emb,
                                         scope="dec_pos",
                                         reuse=tf.AUTO_REUSE,
                                         with_t=True
            )

            t_revs, pos_emb_table_revs = embedding(
                                         tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq_revs)[1]), 0), [tf.shape(self.input_seq_revs)[0], 1]),
                                         vocab_size=args.maxlen,
                                         num_units=args.hidden_units,
                                         zero_pad=False,
                                         scale=False,
                                         l2_reg=args.l2_emb,
                                         scope="dec_pos",
                                         reuse=tf.AUTO_REUSE,
                                         with_t=True
            )

            self.seq += t
            self.seq_revs += t_revs
            self.debug = tf.equal(self.seq, self.seq_revs)#tf.nn.embedding_lookup(item_emb_table, self.input_seq_revs))
            #Dropout and Mask
            with tf.variable_scope('my_dropout_layer'):
                self.seq = tf.layers.dropout(self.seq,
                                            rate=args.dropout_rate,
                                            training=tf.convert_to_tensor(self.is_training), name='my_dropout')
            
            with tf.variable_scope('my_dropout_layer', reuse=True):
                if args.reversed == 1:
                    self.seq_revs = tf.layers.dropout(self.seq_revs,
                                                    rate=args.dropout_rate,
                                                    training=tf.convert_to_tensor(self.is_training), name='my_dropout') 
            
            self.seq *= mask #[B, L, D]
            self.seq_revs *= mask #[B, L, D]
            # self.debug = tf.equal(self.seq, self.seq_revs)
## start feed to network
            
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
                    
                    
                    # Feed forward
                    self.seq = feedforward(normalize(self.seq), num_units=[args.hidden_units, args.hidden_units])
                    self.seq *= mask

                    # Reverse Feed forward
                    self.seq_revs = feedforward(normalize(self.seq_revs), num_units=[args.hidden_units, args.hidden_units])
                    self.seq_revs *= mask
                    
            self.seq = normalize(self.seq) #[B, L, D]
            self.seq_revs = normalize(self.seq_revs) #[B, L, D]
            
        #reshape the matrix
        pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen]) #[B*L]
        neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen]) #[B*L]
        pos_emb = tf.nn.embedding_lookup(item_emb_table, pos) #[B*L, D]
        neg_emb = tf.nn.embedding_lookup(item_emb_table, neg) #[B*L, D]
        seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units]) #[B*L, D]

        pos_revs = tf.reshape(pos_revs, [tf.shape(self.input_seq_revs)[0] * args.maxlen]) #[B*L]
        pos_emb_revs = tf.nn.embedding_lookup(item_emb_table, pos_revs) #[B*L, D]
        seq_emb_revs = tf.reshape(self.seq_revs, [tf.shape(self.input_seq_revs)[0] * args.maxlen, args.hidden_units]) #[B*L, D]
        
        #calculate the predict probability of all items
        test_item_emb = item_emb_table #[item_num, D]
        self.test_logits = tf.matmul(seq_emb, tf.transpose(test_item_emb)) #[B*L, item_num]
        self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], args.maxlen, itemnum+1]) ##[B, L, item_num]
        self.test_logits = self.test_logits[:, -1, :] #[B, item_num]

        # prediction layer
        self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1) #[B*L, D] -> [B*L] element-wise multiply to obtain the similarity of two vectors
        self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
                    
        self.pos_logits_revs = tf.reduce_sum(pos_emb_revs * seq_emb_revs, -1) #[B*L, D] -> [B*L] element-wise multiply to obtain the similarity of two vectors

        # ignore padding items (0) and calculate loss
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen]) #[B*L]
        self.loss_left = tf.reduce_sum(- tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget - tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)
        
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.loss = self.loss_left + sum(reg_losses) 

        variables_net = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        net_save(variables_net, './network_param.txt')

        if args.reversed == 1:
            self.loss_right = tf.reduce_sum(- tf.log(tf.sigmoid(self.pos_logits_revs) + 1e-24) * istarget - tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)
            tf.summary.scalar('loss_right', self.loss_right)
            # self.loss_left *= 0.5
            self.loss_right *= 1
            self.loss = self.loss_left + sum(reg_losses) + self.loss_right

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('loss_left', self.loss_left)

        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)

        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta2=0.98)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def predict(self, sess, u, seq):
        return sess.run(self.test_logits,
                        {self.u: u, self.input_seq: seq, self.is_training: False})

import os
import time
import argparse
import tensorflow as tf
from tqdm import tqdm
import traceback, sys
import json

# self achieve
from sampler import WarpSampler
from model import Model
from util import *

# argument
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--evalnegsample', default=-1, type=int)
parser.add_argument('--reversed', default=0, type=int)
parser.add_argument('--reversed_gen_number', default=-1, type=int)
parser.add_argument('--M', default=10, type=int, help='threshold of augmenation')
parser.add_argument('--reversed_pretrain', default=-1, type=int,
                    help='indicate whether reversed-pretrained model existing, -1=no and 1=yes')
parser.add_argument('--aug_traindata', default=-1, type=int)
args = parser.parse_args()

# load dataset
if not os.path.isdir('./aug_data/' + args.dataset):
    os.makedirs('./aug_data/' + args.dataset)

dataset = data_load(args.dataset, args)
[user_train, user_valid, user_test, original_train, usernum, itemnum] = dataset
sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

num_batch = int(len(user_train) / args.batch_size)
# seqlen and config info
cc = []
for u in user_train:
    cc.append(len(user_train[u]))  # dict[key]=value
cc = np.array(cc)
print('average sequence length: %.2f' % np.mean(cc))
print('min seq length: %.2f' % np.min(cc))
print('max seq length: %.2f' % np.max(cc))
print('quantile 25 percent: %.2f' % np.quantile(cc, 0.25))
print('quantile 50 percent: %.2f' % np.quantile(cc, 0.50))
print('quantile 75 percent: %.2f' % np.quantile(cc, 0.75))

config_signature = 'lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}'.format(
    args.lr,
    args.maxlen,
    args.hidden_units,
    args.num_blocks,
    args.dropout_rate,
    args.l2_emb,
    args.num_heads)

model_signature = '{}_gen_num_{}'.format(config_signature, 5)

aug_data_signature = './aug_data/{}/{}_gen_num_{}_M_{}'.format(
    args.dataset,
    config_signature,
    args.reversed_gen_number,
    args.M)
print(aug_data_signature)

## create a new session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.Session(config=config)

# load model
model = Model(usernum, itemnum, args)
saver = tf.train.Saver()

if args.reversed == 1:  # start pretain and initialize parameters
    sess.run(tf.global_variables_initializer())
else:
    saver.restore(sess, './reversed_models/' + args.dataset + '_reversed/' + model_signature + '.ckpt')
    print('pretain model loaded')

T = 0.0  # account the all training time cost
t0 = time.time()
try:
    for epoch in range(1, args.num_epochs + 1):
        # train model stage
        print('#epoch %d/%d:' % (epoch, args.num_epochs))
        for step in tqdm(range(num_batch)):
            u, seq, pos, neg, seq_revs, pos_revs = sampler.next_batch()  # [b, 1], [b, max_len], [b, max_len], [b, max_len]
            # print(seq[0])
            # print('reverse=============', seq_revs[0])
            # input('check')
            # print(pos[0])
            # print('reverse=============', pos_revs[0])
            # input('check data')
            if args.reversed == 1:
                auc, loss_left, loss_right, loss, debug, _ = sess.run(
                    [model.auc, model.loss_left, model.loss_right, model.loss, model.debug, model.train_op],
                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg, model.input_seq_revs: seq_revs,
                     model.pos_revs: pos_revs,
                     model.is_training: True})  # obtain scalar value
                # print(debug)
                # input('check')
            else:
                auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                        {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                         model.input_seq_revs: seq_revs, model.pos_revs: pos_revs,
                                         model.is_training: True})  # obtain scalar value
        print('loss left: %8f  loss right: %8f  loss: %8f' % (
        loss_left, loss_right, loss)) if args.reversed == 1 else print('loss: %8f' % loss)

        # test model stage
        if (epoch % 20 == 0 and epoch >= 200) or epoch == args.num_epochs:
            t1 = time.time() - t0
            T += t1

            print("start testing...")
            t_test, t_test_short_seq, t_test_short7_seq, \
            t_test_short37_seq, t_test_medium3_seq, t_test_medium7_seq, \
            t_test_long_seq, test_rankitems = evaluate(model, dataset, args, sess, "test")

            # RANK_RESULTS_DIR = f"./rank_results/{args.dataset}_pretain_{args.reversed_pretrain}"
            # if not os.path.isdir(RANK_RESULTS_DIR):
            #     os.makedirs(RANK_RESULTS_DIR)
            # rank_test_file = RANK_RESULTS_DIR + '/' + model_signature + '_predictions.json'
            #if args.reversed == 0:
            #    with open(rank_test_file, 'w') as f:
            #        for eachpred in test_rankitems:
            #            f.write(json.dumps(eachpred) + '\n')

            if not (args.reversed == 1):  # for augmented dataset
                t_valid, t_valid_short_seq, t_valid_short7_seq, \
                t_valid_short37_seq, t_valid_medium3_seq, t_valid_medium7_seq, \
                t_valid_long_seq, valid_rankitems = evaluate(model, dataset, args, sess, "valid")

                print('epoch: ' + str(epoch) + ' validationall: ' + str(t_valid) + '\nepoch: ' + str(
                    epoch) + ' testall: ' + str(t_test))
                print('epoch: ' + str(epoch) + ' validationshort: ' + str(t_valid_short_seq) + '\nepoch: ' + str(
                    epoch) + ' testshort: ' + str(t_test_short_seq))
                print('epoch: ' + str(epoch) + ' validationshort7: ' + str(t_valid_short7_seq) + '\nepoch: ' + str(
                    epoch) + ' testshort7: ' + str(t_test_short7_seq))
                print('epoch: ' + str(epoch) + ' validationshort37: ' + str(t_valid_short37_seq) + '\nepoch: ' + str(
                    epoch) + ' testshort37: ' + str(t_test_short37_seq))
                print('epoch: ' + str(epoch) + ' validationmedium3: ' + str(t_valid_medium3_seq) + '\nepoch: ' + str(
                    epoch) + ' testmedium3: ' + str(t_test_medium3_seq))
                print('epoch: ' + str(epoch) + ' validationmedium7: ' + str(t_valid_medium7_seq) + '\nepoch: ' + str(
                    epoch) + ' testmedium7: ' + str(t_test_medium7_seq))
                print('epoch: ' + str(epoch) + ' validationlong: ' + str(t_valid_long_seq) + '\nepoch: ' + str(
                    epoch) + ' testlong: ' + str(t_test_long_seq))
            else:
                print('epoch: ' + str(epoch) + ' test: ' + str(t_test))

            t0 = time.time()
except Exception as e:
    print(e)
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback,
                              limit=2, file=sys.stdout)
    sampler.close()
    exit(1)

# predict aug-items and save pretrain model
if args.reversed == 1:
    print('start data augmentation...')
    augmented_data = data_augment(model, dataset, args, sess,
                                  args.reversed_gen_number)  # reversed_gen_number equals top_k number.
    with open(aug_data_signature + '.txt', 'w') as f:
        for u, aug_ilist in augmented_data.items():  # dict: key:value
            for ind, aug_i in enumerate(aug_ilist):
                f.write(str(u - 1) + '\t' + str(aug_i - 1) + '\t' + str(-(ind + 1)) + '\n')

    print('augmentation finished!')
    if args.reversed_gen_number > 0:
        if not os.path.exists('./reversed_models/' + args.dataset + '_reversed/'):
            os.makedirs('./reversed_models/' + args.dataset + '_reversed/')
        saver.save(sess, './reversed_models/' + args.dataset + '_reversed/' + model_signature + '.ckpt')
sampler.close()
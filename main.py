# @Time   : 2022/04/12
# @Author : Juyong Jiang
# @Email  : csjuyongjiang@gmail.com

import os
import time
import argparse
import tensorflow as tf
from tqdm import tqdm
import traceback, sys
import json
import numpy as np

# self achieve
from sampler import WarpSampler
from model import Model
from util import set_color, data_load, data_augment, evaluate, record_loss

'''
== Pre-train (Beauty & Phones)
python -u main.py --dataset=Beauty \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.7 --evalnegsample=100 \
                  --hidden_units=128 --num_blocks=2 --num_heads=4 \
                  --reversed=1 --reversed_gen_num=20 --M=20 \
                  --lambda_coef=1.0 \
                  2>&1 | tee pre_train_beauty.log

python -u main.py --dataset=Cell_Phones_and_Accessories \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.5 --evalnegsample=100 \
                  --hidden_units=32 --num_blocks=2 --num_heads=2 \
                  --reversed=1 --reversed_gen_num=20 --M=20 \ 
                  --lambda_coef=1.0 \
                  2>&1 | tee pre_train_phones.log

== Fine-tuning (Beauty & Phones)
python -u main.py --dataset=Beauty \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.7 --evalnegsample=100 \
                  --hidden_units=128 --num_blocks=2 --num_heads=4 \
                  --reversed_pretrain=1 --aug_traindata=15 --M=18 \
                  --alpha_coef=1.0 --clip_k=12 \
                  2>&1 | tee fine_tune_beauty.log

python -u main.py --dataset=Cell_Phones_and_Accessories \
                  --lr=0.001 --maxlen=100 --dropout_rate=0.5 --evalnegsample=100 \
                  --hidden_units=32 --num_blocks=2 --num_heads=2 \
                  --reversed_pretrain=1 --aug_traindata=17 --M=18 \
                  --alpha_coef=0.2 --clip_k=12 \
                  2>&1 | tee fine_tune_phones.log
'''

# argument
def get_arguments():
    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--dataset', default='Beauty', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--evalnegsample', default=-1, type=int)
    parser.add_argument('--del_num', default=0, type=int)
    parser.add_argument('--loss_file', default='fine_tuning_loss.txt', type=str)
    # model
    parser.add_argument('--hidden_units', default=50, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    # hyper
    parser.add_argument('--lambda_coef', default=1.0, type=float)
    parser.add_argument('--alpha_coef', default=1.0, type=float)
    parser.add_argument('--clip_k', default=12, type=int)
    # aug
    parser.add_argument('--reversed', default=0, type=int)
    parser.add_argument('--reversed_gen_number', default=-1, type=int)
    parser.add_argument('--M', default=50, type=int, help='threshold of augmenation')
    parser.add_argument('--reversed_pretrain', default=-1, type=int, help='indicate whether reversed-pretrained model existing, -1=no and 1=yes')
    parser.add_argument('--aug_traindata', default=-1, type=int)
    parser.add_argument('--pre_trained', default=1, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arguments()
    # load augmented dataset
    if not os.path.exists(os.path.join('./aug_data', args.dataset)):
        os.makedirs(os.path.join('./aug_data', args.dataset))
    # loss file
    args.loss_file = args.loss_file.split('.txt')[0] + '_' + args.dataset + '.txt'
    if os.path.exists(args.loss_file):
        os.remove(args.loss_file)
        print('Recreate {}'.format(args.loss_file))

    dataset = data_load(args.dataset, args) 
    [user_train, user_valid, user_test, original_train, usernum, itemnum] = dataset # {uind:[iind,...],...}
    # (user, seq, pos, neg, seq_revs, pos_revs) => one batch: [(), (), ...]
    sampler = WarpSampler(args.del_num, user_train, original_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)
    num_batch = int(len(user_train) / args.batch_size)
    
    # seqlen and config info
    cc = []
    for u in user_train:
        cc.append(len(user_train[u]))  # len statis
    cc = np.array(cc)
    print('average sequence length: %.2f' % np.mean(cc))
    print('min seq length: %.2f' % np.min(cc)) # 3
    print('max seq length: %.2f' % np.max(cc)) 
    print('quantile 25 percent: %.2f' % np.quantile(cc, 0.25))
    print('quantile 50 percent: %.2f' % np.quantile(cc, 0.50))
    print('quantile 75 percent: %.2f' % np.quantile(cc, 0.75))

    # signature
    config_signature = 'lr_{}_maxlen_{}_hsize_{}_nblocks_{}_drate_{}_l2_{}_nheads_{}'.format(
                        args.lr,
                        args.maxlen,
                        args.hidden_units,
                        args.num_blocks,
                        args.dropout_rate,
                        args.l2_emb,
                        args.num_heads)

    model_signature = '{}_gen_num_{}'.format(config_signature, 50)

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

    if args.reversed == 1:  # start pre-taining and initialize parameters
        sess.run(tf.global_variables_initializer())
    elif args.pre_trained == 1: # whether use pre-trained model or train model from scratch
        saver.restore(sess, './reversed_models/' + args.dataset + '_reversed/' + model_signature + '.ckpt')
        print('Pretain model loaded {}......'.format('./reversed_models/' + args.dataset + '_reversed/' + model_signature + '.ckpt'))
    else:
        sess.run(tf.global_variables_initializer())
        args.alpha_coef = 0.0
        print('Training model from scratch......')

    T = 0.0  # account the all training time cost
    t0 = time.time()
    try:
        for epoch in range(1, args.num_epochs + 1):
            # train model stage
            # print('Epoch %d/%d:' % (epoch, args.num_epochs))
            for step in tqdm(range(num_batch), total=num_batch, ncols=100, desc=set_color(f"Train {epoch}/{args.num_epochs}", 'pink'),):
                u, seq, pos, neg, seq_revs, pos_revs, seq_origin, pos_origin, neg_origin = sampler.next_batch()  # [b, 1], [b, max_len], [b, max_len], [b, max_len], [b, max_len], [b, max_len]
                # print('seq =============', seq[0][-10:])
                # print('seq reverse=============', seq_revs[0][-10:])
                # print('seq origin=============',  seq_origin[0][-10:])
                # print("")
                # print('pos =============', pos[0][-10:])
                # print('pos reverse=============', pos_revs[0][-10:])
                # print('pos origin=============',  pos_origin[0][-10:])
                # print("")
                # print('neg =============', neg[0][-10:])
                # print('neg origin=============',  neg_origin[0][-10:])
                # input('check data')
                if args.reversed == 1:
                    auc, loss_left, loss_right, loss, debug, _ = sess.run(
                                                                [model.auc, 
                                                                model.loss_left, 
                                                                model.loss_right, 
                                                                model.loss, 
                                                                model.debug, 
                                                                model.train_op],

                                                                {model.u: u, 
                                                                model.input_seq: seq, 
                                                                model.pos: pos, 
                                                                model.neg: neg, 
                                                                model.input_seq_revs: seq_revs,
                                                                model.pos_revs: pos_revs,
                                                                model.input_seq_origin: seq_origin,
                                                                model.pos_origin: pos_origin,
                                                                model.neg_origin: neg_origin,                                                                 
                                                                model.is_training: True})  # obtain scalar value
                    # print(debug)
                    # input('check')
                else:
                    auc, loss_left, kl_loss, loss, debug, embs_table, _ = sess.run(
                                  [model.auc, 
                                   model.loss_left, 
                                   model.kl_loss, 
                                   model.loss,
                                   model.debug,
                                   model.item_emb_table,
                                   model.train_op],

                                  {model.u: u, 
                                   model.input_seq: seq, 
                                   model.pos: pos, 
                                   model.neg: neg,
                                   model.input_seq_revs: seq_revs, 
                                   model.pos_revs: pos_revs,
                                   model.input_seq_origin: seq_origin,
                                   model.pos_origin: pos_origin,
                                   model.neg_origin: neg_origin,  
                                   model.is_training: True})  # obtain scalar value
                    np.save('{}_bicat_embs'.format(args.dataset), embs_table)
            # print(kl_loss)
            # print(loss_aug)
            # print(loss_origin)
            # print(loss)
            # input('check')                      
            print('loss left: %8f  loss right: %8f loss: %8f' % (loss_left, loss_right, loss)) if args.reversed == 1 else \
            print('loss_left: %8f kl_loss: %8f loss: %8f' % (loss_left, kl_loss, loss))
            # print(debug)
            # input('check')
            # record various loss
            if args.reversed_pretrain == 1:
                record_loss(args.loss_file, loss_left, kl_loss, loss)

            # test model stage
            if (epoch % 20 == 0 and epoch >= 200) or epoch == args.num_epochs:
                t1 = time.time() - t0
                T += t1

                print("start testing...")
                t_test, t_test_short_seq, t_test_short37_seq, \
                t_test_short720_seq, t_test_medium2050_seq, \
                t_test_long_seq, test_rankitems = evaluate(model, dataset, args, sess, "test")

                # RANK_RESULTS_DIR = f"./rank_results/{args.dataset}_pretain_{args.reversed_pretrain}"
                # if not os.path.isdir(RANK_RESULTS_DIR):
                #     os.makedirs(RANK_RESULTS_DIR)
                # rank_test_file = RANK_RESULTS_DIR + '/' + model_signature + '_predictions.json'
                #if args.reversed == 0:
                #    with open(rank_test_file, 'w') as f:
                #        for eachpred in test_rankitems:
                #            f.write(json.dumps(eachpred) + '\n')

                if not (args.reversed == 1):  # fine-turing 
                    t_valid, t_valid_short_seq, t_valid_short37_seq, \
                    t_valid_short720_seq, t_valid_medium2050_seq, \
                    t_valid_long_seq, valid_rankitems = evaluate(model, dataset, args, sess, "valid")

                    print('epoch: ' + str(epoch) + ' testall: ' + str(t_test))
                    print('epoch: ' + str(epoch) + ' testshort: ' + str(t_test_short_seq))
                    print('epoch: ' + str(epoch) + ' testshort37: ' + str(t_test_short37_seq))
                    print('epoch: ' + str(epoch) + ' testshort720: ' + str(t_test_short720_seq))
                    print('epoch: ' + str(epoch) + ' testmedium2050: ' + str(t_test_medium2050_seq))
                    print('epoch: ' + str(epoch) + ' testlong: ' + str(t_test_long_seq))
                else:
                    print('epoch: ' + str(epoch) + ' test: ' + str(t_test)) # it is wrong and not necessary

                t0 = time.time()
    except Exception as e:
        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
        sampler.close()
        exit(1)

    # predict aug-items and save pretrained model
    if args.reversed == 1:
        print('start data augmentation...')
        augmented_data = data_augment(model, dataset, args, sess, args.reversed_gen_number)  # reversed_gen_number equals top_k number.
        with open(aug_data_signature + '.txt', 'w') as f:
            for u, aug_ilist in augmented_data.items():  
                for ind, aug_i in enumerate(aug_ilist):
                    f.write(str(u - 1) + '\t' + str(aug_i - 1) + '\t' + str(-(ind + 1)) + '\n') # by time order (early -> later)

        print('augmentation finished!')
        if args.reversed_gen_number > 0:
            if not os.path.exists(os.path.join('./reversed_models', args.dataset + '_reversed/')):
                os.makedirs(os.path.join('./reversed_models', args.dataset + '_reversed/'))
            saver.save(sess, os.path.join('./reversed_models', args.dataset + '_reversed', model_signature + '.ckpt'))
        print("reversed models saved!")
    sampler.close()
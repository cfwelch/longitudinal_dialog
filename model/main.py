

import msgpack, math, pdb, os, inspect, sys, time, datetime
import torch
import numpy as np
from torch.autograd import Variable
import models
from dataloader import Dataloader
# import visdom
from argparse import ArgumentParser
from torch.nn.utils import clip_grad_norm

from utils import settings, gprint, label_set, time_labels, tag_set, EMBEDDING_SIZE, NUMBER_OF_PEOPLE, RA_IND, FAM_IND, SCC_IND, SG_IND, SCH_IND, WORK_IND, NPR_IND, uax_set, uv_to_parts, uax_names

parser = ArgumentParser()
# feature flags
parser.add_argument('-aur', '--add-user-representation', dest='add_user_representation', help='Include user representation features', default=False, action='store_true')
parser.add_argument('-als', '--add-liwc-values', dest='add_liwc_sum', help='Include LIWC features', default=False, action='store_true')
parser.add_argument('-atv', '--add-time-values', dest='add_time_values', help='Include time features', default=False, action='store_true')
parser.add_argument('-afq', '--add-frequency-values', dest='add_freq_values', help='Include frequency values in model', default=False, action='store_true')
parser.add_argument('-atc', '--add-turn-change-values', dest='add_tc_values', help='Include turn change featuers', default=False, action='store_true')
parser.add_argument('-asv', '--add-stopword-values', dest='add_stop_values', help='Include stopword features', default=False, action='store_true')
parser.add_argument('-alv', '--add-lsm-values', dest='add_lsm_values', help='Include linguistic style matching features', default=False, action='store_true')
parser.add_argument('-asu', '--add-surface-values', dest='add_surf_values', help='Include surface text features', default=False, action='store_true')
parser.add_argument('-agv', '--add-graph-values', dest='add_graph_values', help='Include graph distance features', default=False, action='store_true')
# longitudinal research flags
parser.add_argument('-cv', '--cont-value', dest='continuous', help='Problem formulation is continuous', default=False, action='store_true')
parser.add_argument('-rt', '--response-time', dest='response_time', help='Problem formulation is response time', default=False, action='store_true')
parser.add_argument('-ua', '--user-attribute', dest='user_attributes', help='Problem formulation is user attributes', default=False, action='store_true')
parser.add_argument('-sa', '--single-attribute', dest='single_attribute', help='Do not jointly learn attributes, instead just this one', default=None, type=str)
# general deep learning flags
parser.add_argument('-rtype', '--rtype', dest='rtype', help='Run type can be \'train\' or \'test\'', default='train', type=str)
parser.add_argument('-s', '--show', dest='show', help='Default graph epoch progress with Visdom', default=True, action='store_false')
parser.add_argument('-v', '--verbose', dest='verbose', help='Add verbose error analysis output', default=False, action='store_true')
parser.add_argument('-ea', '--error-analysis', dest='error_analysis', help='Writes the errors for each evaluation after each epoch', default=False, action='store_true')
parser.add_argument('-lr', '--learning-rate', dest='lr', help='Learning Rate', default=1e-4, type=float)
parser.add_argument('-e', '--num-epochs', dest='epochs', help='Number of epochs', default=20, type=int)#30
parser.add_argument('-ss', '--start-split', dest='start_split', help='At split for dataloader ua', default=0, type=int)
parser.add_argument('-b', '--batch-size', dest='batch_size', help='Mini-batch size', default=64, type=int)
parser.add_argument('-hs', '--hidden-size', dest='hidden_size', help='Size of hidden to model', default=-1, type=int)
parser.add_argument('-d', '--dropout', dest='dropout', help='Dropout', default=0.5, type=float)
parser.add_argument('-wd', '--wdecay', dest='w_decay', help='Weight Decay', default=0, type=float)
parser.add_argument('--grad-clip', dest='grad_clip', help='Gradient clipping', default=1, type=float)
parser.add_argument('-cpu', '--use-cpus', dest='use_cpus', help='Use GPUs by default but use CPUs if this flag is set.', default=False, action='store_true')
parser.add_argument('--freeze-embeds', dest='freeze_embeds', help='Freze embeddings of pre-trained tokens', default=0, type=int)
parser.add_argument('--patience', dest='patience', help='Number of epochs to train before early stopping', default=3, type=int)#5
opt = parser.parse_args()

total_scores_ua = [0.0]*len(uax_set)
total_guesses_ua = [[] for i in range(len(uax_set))]
keys = list(tag_set.keys())
keys.sort()

single_att = -1
if opt.single_attribute != None and opt.single_attribute not in uax_names:
    print('Single attribute parameter must be in: ' + str(uax_names))
    sys.exit(0)
elif opt.single_attribute != None:
    opt.user_attributes = True
    single_att = uax_names.index(opt.single_attribute)

def cprint(msg, error=False, important=False):
    gprint(msg, ('single_att_' + opt.single_attribute + '_' if opt.single_attribute != None else '') + 'model' + ('_ur' if opt.add_user_representation else '') + ('_ls' if opt.add_liwc_sum else '') + ('_tv' if opt.add_time_values else '') + ('_fq' if opt.add_freq_values else '') + ('_tc' if opt.add_tc_values else '') + ('_sv' if opt.add_stop_values else '') + ('_lv' if opt.add_lsm_values else '') + ('_su' if opt.add_surf_values else '') + ('_gv' if opt.add_graph_values else '') + '_' + opt.rtype, error, important)

# vis = visdom.Visdom()
hidden_list = [512] # 32, 64, 128, 256
if opt.hidden_size > 0:
    hidden_list = [opt.hidden_size]

for dirname in ['logs', 'errors']:
    if not os.path.exists(dirname):
        os.makedirs(dirname)

results_dict = {hidd: None for hidd in hidden_list}

#VOCAB_FILE = 'vocabulary'
#REVERSE_VOCAB_FILE = 'reverse_vocabulary'
embed_prefix = ''
EMBEDS_FILE = embed_prefix + 'embeds.th'

cprint('File settings : ', important=True)
#cprint('\tVocaulary file: ' + VOCAB_FILE)
#cprint('\tReverse Vocaulary file: ' + REVERSE_VOCAB_FILE)
cprint('\tEmbeddings file: ' + EMBEDS_FILE)

#with open(FILE_PATH + VOCAB_FILE, 'rb') as f:
#    t_vocab = msgpack.unpackb(f.read())
#    vocab = {}
#    for key in t_vocab:
#        vocab[key.decode()] = t_vocab[key]
#    cprint('\tVocab Length: ' + str(len(vocab)))

crit = torch.nn.CrossEntropyLoss() if not opt.continuous else torch.nn.L1Loss()
evalSM = torch.nn.Softmax()
if not opt.use_cpus:
    crit = crit.cuda()
    evalSM = evalSM.cuda()

def get_train_accuracy(logits, labels):
    if not opt.continuous:
        if opt.user_attributes:
            acc_avg = 0.0
            if opt.single_attribute != None:
                probs = evalSM(logits)
                inds = probs.max(1)[1]
                return get_accuracy(inds, labels[single_att], uatt=single_att)
            else:
                for iii in range(len(logits)):
                    probs = evalSM(logits[iii])
                    inds = probs.max(1)[1]
                    acc_avg += get_accuracy(inds, labels[iii], uatt=iii)
                return acc_avg / len(logits)
        else:
            probs = evalSM(logits)
            inds = probs.max(1)[1]
            return get_accuracy(inds, labels)
    else:
        #print('logits are: ' + str(logits))
        #print('labels are: ' + str(labels))
        return crit(logits, labels).data[0]

def get_accuracy(inds, labels, uatt=None, param_str=None, evaluate=False, test=False):
    #print('get accuracy inds: ' + str(inds))
    #print('get accuracy labels: ' + str(labels))
    #print('get accuracy uatt: ' + str(uatt))
    if opt.user_attributes:
        uatt_key = uax_set[uatt]
        ga_lbs = tag_set[keys[uatt_key]] if uatt_key != SCC_IND else tag_set[keys[uatt_key]][:2]
    else:
        ga_lbs = label_set if not opt.response_time else time_labels
    acc = 0.0
    error_str = ''
    nontorch = 0.0
    nones = 0
    totals = {label_name: 0 for label_name in ga_lbs}
    correct = {label_name: 0 for label_name in ga_lbs}
    preds = {label_name: 0 for label_name in ga_lbs}
    if not opt.response_time and not opt.user_attributes:
        totals['other'] = 0
        correct['other'] = 0
        preds['other'] = 0
    for _i in range(0, len(inds.data)):
        if not opt.response_time:
            pred_name = ga_lbs[inds.data[_i]] if inds.data[_i] < len(ga_lbs) else 'other'
        else:
            pred_name = ga_lbs[inds.data[_i]]
        preds[pred_name] += 1.0

        if not opt.response_time:
            label_name = ga_lbs[labels.data[_i]] if labels.data[_i] < len(ga_lbs) else 'other'
        else:
            label_name = ga_lbs[labels.data[_i]]

        #cprint('label: ' + str(label_name) + ' _ pred: ' + str(pred_name), error=True)

        if inds.data[_i] != labels.data[_i]:
            if pred_name == None or label_name == None:
                cprint('ERROR: label: ' + str(label_name) + ' _ pred: ' + str(pred_name), error=True)
                nones += 1
        else:
            correct[label_name] += 1.0
            nontorch += 1.0
        totals[label_name] += 1.0
    nt_acc = nontorch / len(inds.data)
    if evaluate:
        if opt.user_attributes:
            cprint('User Attribute ' + keys[uatt_key], important=True)
        cprint(str(int(nontorch)) + ' correct out of ' + str(len(inds.data)), error=True)
        most_guessed = -1
        most_guessed_ind = -1
        for key in correct:
            cprint('\t' + '{:>10}'.format(key) + ': ' + '{:4d}'.format(int(correct[key])) + '/' + '{:4d}'.format(int(totals[key])) + ' = ' + '{:2.3f}'.format(correct[key]*100.0/totals[key] if totals[key] > 0 else 0))
            if preds[key] > most_guessed:
                most_guessed = preds[key]
                most_guessed_ind = key
        if test:
            cprint('Guessing that person has value ' + most_guessed_ind + ' for key ' + keys[uatt_key], important=True)
    acc = nt_acc
    # write errors to file
    if evaluate and opt.error_analysis:
        with open('errors/' + param_str, 'w') as handle:
            handle.write(error_str)
    return acc*100 if not opt.user_attributes or not test else ga_lbs.index(most_guessed_ind)

def evaluate_model(net, param_str, split='dev'):
    cprint('Evaluating model on ' + split + ' split...')
    net.eval()
    eparset = dl.get_evaluation_data(split)
    cprint('Done getting the evaluation data...')

    if split == 'dev':
        eparset = [eparset]

    uatt_set_ppl = []
    with torch.no_grad():
        for eval_params in eparset:
            if opt.user_attributes and opt.single_attribute == None:
                outs = [[] for i in range(7)]
                labels = [[] for i in range(7)]
            else:
                outs = []
                labels = []
            for i in range(0, len(eval_params)):
                batch = Variable(eval_params[i][0].cuda()) if not opt.use_cpus else Variable(eval_params[i][0])
                rev_batch = Variable(eval_params[i][1].cuda()) if not opt.use_cpus else Variable(eval_params[i][1])
                lens = Variable(eval_params[i][3].cuda()) if not opt.use_cpus else Variable(eval_params[i][3])
                user_rep = Variable(eval_params[i][4].cuda()) if not opt.use_cpus else Variable(eval_params[i][4])
                liwc_sum = Variable(eval_params[i][5].cuda()) if not opt.use_cpus else Variable(eval_params[i][5])
                time_values = Variable(eval_params[i][6].cuda()) if not opt.use_cpus else Variable(eval_params[i][6])
                freq_values = Variable(eval_params[i][7].cuda()) if not opt.use_cpus else Variable(eval_params[i][7])
                tc_values = Variable(eval_params[i][8].cuda()) if not opt.use_cpus else Variable(eval_params[i][8])
                stop_values = Variable(eval_params[i][9].cuda()) if not opt.use_cpus else Variable(eval_params[i][9])
                lsm_values = Variable(eval_params[i][10].cuda()) if not opt.use_cpus else Variable(eval_params[i][10])
                surface_values = Variable(eval_params[i][11].cuda()) if not opt.use_cpus else Variable(eval_params[i][11])
                graph_values = Variable(eval_params[i][12].cuda()) if not opt.use_cpus else Variable(eval_params[i][12])
                tout = net((batch, rev_batch), lens, user_rep, liwc_sum, time_values, freq_values, tc_values, stop_values, lsm_values, surface_values, graph_values)
                # get labels
                if opt.user_attributes:
                    if opt.single_attribute == None:
                        for ii in range(len(eval_params[i][2])):
                            labels[ii].append(eval_params[i][2][ii])
                            outs[ii].append(tout[ii])
                    else:
                        labels.append(eval_params[i][2][single_att])
                        outs.append(tout)
                else:
                    labels.append(eval_params[i][2])
                    outs.append(tout)

            if opt.user_attributes and opt.single_attribute == None:
                outs = [torch.stack(_oq, 0) for _oq in outs]
                out = [_oq.squeeze() for _oq in outs]
                labels = [torch.stack(_lq) for _lq in labels]
                labels = [_lq.squeeze() for _lq in labels]
                labels = [Variable(_lq.cuda()) if not opt.use_cpus else Variable(_lq) for _lq in labels]
            else:
                out = torch.stack(outs, 0)
                out = out.squeeze()
                labels = torch.stack(labels)
                labels = labels.squeeze()
                labels = Variable(labels.cuda()) if not opt.use_cpus else Variable(labels)

            if not opt.continuous:
                if opt.user_attributes:
                    uatt_set = []
                    acc_avg = 0.0
                    if opt.single_attribute != None:
                        probs = evalSM(out)
                        inds = probs.max(1)[1]
                        #print('inds is: ' + str(inds.data.numpy().tolist()))
                        #print('labels is: ' + str(labels.data.numpy().tolist()))
                        #print('singleatt is: ' + str(single_att))
                        if split != 'test':
                            acc_avg += get_accuracy(inds, labels, uatt=single_att, param_str=param_str, evaluate=True, test=False)
                        else:
                            uatt_set.append(get_accuracy(inds, labels, uatt=single_att, param_str=param_str, evaluate=True, test=True))
                    else:
                        for iii in range(len(out)):
                            probs = evalSM(out[iii])
                            inds = probs.max(1)[1]
                            if split != 'test':
                                acc_avg += get_accuracy(inds, labels[iii], uatt=iii, param_str=param_str, evaluate=True, test=False)
                            else:
                                uatt_set.append(get_accuracy(inds, labels[iii], uatt=iii, param_str=param_str, evaluate=True, test=True))
                    uatt_set_ppl.append(uatt_set)
                    if split != 'test':
                        acc = acc_avg / len(out)
                else:
                    prob = evalSM(out)
                    inds = prob.max(1)[1]
                    if opt.response_time:
                        assert inds.data.lt(len(time_labels)+1).sum() == inds.data.ge(0).sum() == len(labels)
                    elif not opt.user_attributes:
                        assert inds.data.lt(len(label_set)+1).sum() == inds.data.ge(0).sum() == len(labels)
                    #acc = torch.eq(inds.data, labels).sum()/float(len(labels))
                    acc = get_accuracy(inds, labels, param_str=param_str, evaluate=True)
            else:
                acc = crit(out, labels).data[0]

    if opt.user_attributes and split == 'test':
        pass
    else:
        cprint('Evaluation (' + split + ') score : %.3f%%' % (acc))
    return acc if not opt.user_attributes or split != 'test' else uatt_set_ppl

dl_epochs = opt.epochs - 1
dl = Dataloader(dl_epochs, opt.batch_size, opt.continuous, opt.response_time, opt.user_attributes, opt.start_split, single_att, embed_prefix=embed_prefix)
runs = []

for hidden in hidden_list:
    dl.serve_data(all_data=(opt.user_attributes or opt.rtype == 'test'))

    while not opt.user_attributes or (opt.user_attributes and dl.serving_person < len(dl.ppl_keys)):
        # doesn't matter if it is UA
        output_size = len(label_set) + 1 if not opt.response_time else len(time_labels)
        cprint('Number of output classes: ' + str(output_size), important=True)

        cprint('\n')
        cprint('*'*40, important=True)
        cprint('Hidden Size : ' + str(hidden), important=True)
        cprint('Dropout: ' + str(opt.dropout), important=True)
        cprint('*'*40, important=True)

        cprint('Creating Model...')
        # load pretrained embeddings from file and freeze them
        embeds = torch.load(EMBEDS_FILE)
        embed_layer = torch.nn.Embedding(embeds.size(0) + 1, EMBEDDING_SIZE, 0)
        embed_layer.weight.data = embeds
        embed_layer.weight.requires_grad = True
        num_non_freeze_inds = 4
        cprint('Number of people: ' + str(NUMBER_OF_PEOPLE))

        net = models.net(hidden, output_size, embed_layer, EMBEDDING_SIZE, opt.dropout, opt.add_user_representation, opt.add_liwc_sum, opt.add_time_values, opt.add_freq_values, opt.add_tc_values, opt.add_stop_values, opt.add_lsm_values, opt.add_graph_values, opt.add_surf_values, dl.liwc_len, dl.time_len, dl.freq_len, dl.tc_len, dl.stop_len, dl.lsm_len, dl.surf_len, NUMBER_OF_PEOPLE, is_cont=opt.continuous, is_ua=opt.user_attributes, s_ua=single_att)
        if not opt.use_cpus:
            net = net.cuda()
        cur_lr = opt.lr
        cprint('Initializing the optimizer with learning rate: ' + str(cur_lr), important=True)
        optimizer = torch.optim.Adam(net.parameters(), lr=cur_lr, weight_decay=opt.w_decay)

        # dl.getFoldInds()
        best_acc = 0 if not opt.continuous else 999999999
        best_epoch = 0
        patience = 1
        train_loss, eval_accs, train_accuracy = [], [], []
        last_epoch = []
        train_averages = []
        test_guess = None

        cprint('Training model with TrainData...')
        while dl.stop_flag is False:
            if dl.iter_ind == 0 and dl.epoch > 0:
                cprint('-'*30 + 'END OF EPOCH : ' + str(dl.epoch) + '-'*30, important=True)

                # Evaluation at end of epoch
                evalAcc = evaluate_model(net, 'epoch_' + str(dl.epoch) + '_hs_' + str(hidden), split=('dev' if opt.rtype == 'train' else 'test'))
                eval_accs.append(evalAcc)
                last_epoch = np.array(last_epoch)
                tavg = np.average(last_epoch)
                train_averages.append(tavg)
                cprint('Average training score: ' + '{:2.3f}'.format(tavg), important=True)
                cprint('Standard deviation: ' + '{:2.3f}'.format(np.std(last_epoch)), important=True)
                last_epoch = []

                if (evalAcc > best_acc and not opt.continuous) or (evalAcc < best_acc and opt.continuous):
                    patience = 1
                    best_acc = evalAcc
                    best_epoch = dl.epoch
                    test_acc = 0
                    if opt.user_attributes:
                        cprint('Best development accuracy so far. Checking test...')
                        test_guess = evaluate_model(net, 'epoch_' + str(dl.epoch) + '_hs_' + str(hidden), split='test')
                    results_dict[hidden] = (best_acc, best_epoch, test_acc)
                else:
                    patience += 1
                    cprint('No improvement in accuracy... Patience increased to: ' + str(patience))
                    if patience > opt.patience and opt.rtype == 'train':
                        cprint('Exceeded patience limit! Stopping Training...')
                        break
                net.train(mode=True)

            batch, rev_batch, labels, lens, user_rep, liwc_sum, time_values, freq_values, tc_values, stop_values, lsm_values, surface_values, graph_values = dl.get_train_batch()
            if batch is None and rev_batch is None and labels is None:
                break

            # standard training 
            batch = Variable(batch.cuda()) if not opt.use_cpus else Variable(batch)
            rev_batch = Variable(rev_batch.cuda()) if not opt.use_cpus else Variable(rev_batch)
            lens = Variable(lens.cuda()) if not opt.use_cpus else Variable(lens)
            user_rep = Variable(user_rep.cuda()) if not opt.use_cpus else Variable(user_rep)
            liwc_sum = Variable(liwc_sum.cuda()) if not opt.use_cpus else Variable(liwc_sum)
            time_values = Variable(time_values.cuda()) if not opt.use_cpus else Variable(time_values)
            freq_values = Variable(freq_values.cuda()) if not opt.use_cpus else Variable(freq_values)
            tc_values = Variable(tc_values.cuda()) if not opt.use_cpus else Variable(tc_values)
            stop_values = Variable(stop_values.cuda()) if not opt.use_cpus else Variable(stop_values)
            lsm_values = Variable(lsm_values.cuda()) if not opt.use_cpus else Variable(lsm_values)
            surface_values = Variable(surface_values.cuda()) if not opt.use_cpus else Variable(surface_values)
            graph_values = Variable(graph_values.cuda()) if not opt.use_cpus else Variable(graph_values)
            if opt.user_attributes:
                labels = [Variable(_lq.cuda()) if not opt.use_cpus else Variable(_lq) for _lq in labels]
            else:
                labels = Variable(labels.cuda()) if not opt.use_cpus else Variable(labels)

            out = net((batch, rev_batch), lens, user_rep, liwc_sum, time_values, freq_values, tc_values, stop_values, lsm_values, surface_values, graph_values)

            loss_sum = 0
            if opt.user_attributes:
                if opt.single_attribute == None:
                    tloss_sum = 0
                    for iii in range(len(labels)):
                        loss = crit(out[iii], labels[iii])
                        tloss_sum += loss
                        #loss.backward(retain_graph=(iii < len(labels) - 1))
                        loss_sum += loss.item()
                    tloss_sum.backward()
                else:
                    loss = crit(out, labels[single_att])
                    loss.backward()
                    loss_sum += loss.item()
            else:
                loss = crit(out, labels)
                loss.backward()
                loss_sum = loss.item()
            train_loss.append(loss_sum*1.0/len(labels))
            c_train_acc = get_train_accuracy(out, labels)
            last_epoch.append(c_train_acc)
            train_accuracy.append(c_train_acc)

            #if opt.freeze_embeds:
            #    #set gradient to 0 to freeze them
            #    net.wordEmbed.weight.grad[:-num_non_freeze_inds, :] = 0
            clip_grad_norm(net.parameters(), opt.grad_clip)

            optimizer.step()
            #print(embed_layer(Variable(torch.LongTensor([vocab['hah']]))))
            optimizer.zero_grad()
            dl_total = (len(dl.train_inds) // dl.batch_size) + 1
            modval = dl_total//16 if dl_total > 16 else 1
            if dl.global_ind % modval == 0:
                cprint('Epoch:' + str(dl.epoch + 1) + ' | Batch : ' + str(dl.global_ind) + '/' + str(dl_total) + ' \tLoss: ' + '{:2.3f}'.format(loss_sum) + '\t Accuracy: ' + '{:2.3f}'.format(train_accuracy[-1]))

        #runs.append(eval_accs[-1])
        cprint('Erasing model and gradient parameters. Resetting model...')
        del net, optimizer

        if opt.user_attributes:
            # Check the test acc again and add it to global vars
            tpcounter = 0
            for tpkey in dl.ppl_key_set:
                val_set = uv_to_parts(dl.test_data[tpkey][0][b'label'])
                cprint('Person ' + str(tpkey) + '...', error=True)
                for ik in range(0, len(uax_set)):
                    if opt.single_attribute != None and single_att != ik:
                        continue
                    tkeyname = keys[uax_set[ik]]
                    print('test_guess: ' + str(test_guess[tpcounter]))
                    guess_val = test_guess[tpcounter][ik] if opt.single_attribute == None else test_guess[tpcounter][0]
                    guess_name = tag_set[tkeyname][guess_val]#np.argmax()
                    #print('valset: ' + str(val_set[uax_set[ik]]))
                    #print('valset1: ' + str(val_set[uax_set[ik]].index(1)))
                    actual_name = tag_set[tkeyname][val_set[uax_set[ik]].index(1)]
                    cprint(tkeyname + ': guessed as ' + guess_name + ' but actually is ' + actual_name)
                    total_guesses_ua[ik].append(guess_name)
                    if actual_name == guess_name:
                        total_scores_ua[ik] += 1.0

                cprint('\n')
                cprint('Accuracy so far: ', important=True)
                for ik in range(0, len(uax_set)):
                    cprint('\t' + keys[uax_set[ik]] + ': ' + str(total_scores_ua[ik]) + '/' + str(dl.serving_person + 1) + ' = ' + str(total_scores_ua[ik]*1.0/(dl.serving_person + 1)))

                cprint('all guesses so far: ' + str(total_guesses_ua))

            # go to next person
            dl.serving_person += 1
            dl.next_ua()
            tpcounter += 1

        ## for plotting with FB's Visdom library --- old options
        #title = '%s | %d | %0.2f' % (opt.classification, hidden, opt.dropout)
        env_name = 'Longitudinal Utterance Prediction'
        # Graph with Visdom
        tepochs = range(0, len(train_averages))
        # add epochs
        taxis = np.array([tepochs]*2)
        taxis = taxis.transpose()
        # add accuracies and y-axis
        yaccs = np.array([eval_accs, train_averages])
        yaccs = yaccs.transpose()
        # display values
        disp_lr = '{:1.0e}'.format(opt.lr).split('-')
        disp_lr = disp_lr[0] + '-' + str(int(disp_lr[1]))
        disp_wd = '{:1.0e}'.format(opt.w_decay)
        if opt.w_decay == 0:
            disp_wd = '0'
        elif '-' in disp_wd or '+' in disp_wd:
            disp_wd = disp_wd.split('-')
            disp_wd = disp_wd[0].split('+') if len(disp_wd) == 1 else disp_wd
            disp_wd = disp_wd[0] + '-' + str(int(disp_wd[1]))
        # print title
        # if opt.show:
        #     line_title = 'LR=' + disp_lr + ', WD=' + disp_wd + ',H=' + str(hidden) + ', B=' + str(opt.batch_size) + ', M:' + '{:2.1f}'.format(float(results_dict[hidden][0])) + '@' + str(int(results_dict[hidden][1]))
        #     vis.line(X=taxis, Y=yaccs, opts=dict(legend=['Evaluation', 'Training'], xlabel='Epoch', ylabel='Accuracy', title=line_title), env=env_name)
        #     vis.save([env_name])
        if not opt.user_attributes:
            break

cprint('\n\n')
best_params = [0.0, 0.0, 0.0]
for hidd in hidden_list:
    cprint('Hidden Size : ' + str(hidd), important=True)
    cprint('Best Acc: ' + str(results_dict[hidd][0]) + '\t Epochs: ' + str(results_dict[hidd][1]), important=True)
    if results_dict[hidd][0] > best_params[0]:
        best_params[0] = results_dict[hidd][0]
        best_params[1] = results_dict[hidd][1]
        best_params[2] = hidd
# Best parameter set
cprint('Best parameter set is hidden size ' + str(best_params[2]) + ' with ' + str(best_params[1]) + ' epochs.', important=True)

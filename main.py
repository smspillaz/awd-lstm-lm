import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

import data
import model as model_module

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--load-only', action='store_true', help='Just load the model, don\'t train it')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

def model_save(model, optimizer, criterion, fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)

def model_load(fn):
    with open(fn, 'rb') as f:
        return torch.load(f)

import os
import hashlib

def load_dataset(path, batch_size):
    fn = 'corpus.{}.data'.format(hashlib.md5(path.encode()).hexdigest())
    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)
    else:
        print('Producing dataset...')
        corpus = data.Corpus(path)
        torch.save(corpus, fn)

    eval_batch_size = 10
    test_batch_size = 1
    train_data = batchify(corpus.train, batch_size, args)
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    return corpus, train_data, val_data, test_data, batch_size, eval_batch_size, test_batch_size

###############################################################################
# Build the model
###############################################################################

from splitcross import SplitCrossEntropyLoss
def build_model(corpus,
                model_name,
                emsize,
                nhid,
                nlayers,
                dropout,
                dropouth,
                dropouti,
                dropoute,
                wdrop,
                lr,
                tied,
                resume,
                cuda):
    criterion = None

    ntokens = len(corpus.dictionary)
    model = model_module.RNNModel(model_name,
                                  ntokens,
                                  emsize,
                                  nhid,
                                  nlayers,
                                  dropout,
                                  dropouth,
                                  dropouti,
                                  dropoute,
                                  wdrop,
                                  tied)
    ###
    if resume:
        print('Resuming model ...')
        model, criterion, optimizer = model_load(resume)
        optimizer.param_groups[0]['lr'] = lr
        model.dropouti, model.dropouth, model.dropout, model.dropoute = dropouti, dropouth, dropout, dropoute
        if wdrop:
            from weight_drop import WeightDrop
            for rnn in model.rnns:
                if type(rnn) == WeightDrop: rnn.dropout = wdrop
                elif rnn.zoneout > 0: rnn.zoneout = wdrop
    ###
    if not criterion:
        splits = []
        if ntokens > 500000:
            # One Billion
            # This produces fairly even matrix mults for the buckets:
            # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
            splits = [4200, 35000, 180000]
        elif ntokens > 75000:
            # WikiText-103
            splits = [2800, 20000, 76000]
        print('Using', splits)
        criterion = SplitCrossEntropyLoss(emsize, splits=splits, verbose=False)
    ###
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    ###
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Args:', args)
    print('Model total parameters:', total_params)

    return model, criterion, None

###############################################################################
# Training code
###############################################################################

def evaluate(corpus, model_type, model, data_source, bptt, criterion, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if model_type == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, bptt):
        data, targets = get_batch(data_source, i, bptt, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(corpus, train_data, model_type, model, optimizer, alpha, beta, batch_size, clip, log_interval, epoch, arg_bptt, criterion):
    # Turn on training mode which enables dropout
    params = list(model.parameters()) + list(criterion.parameters())

    if model_type == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = arg_bptt if np.random.random() < 0.95 else arg_bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / bptt
        model.train()
        data, targets = get_batch(train_data, i, bptt, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if alpha: loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if beta: loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if clip: torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3} | {:5}/{:5} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // arg_bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len


def training_loop(corpus, train_data, val_data, model_type, model, lr, epochs, wdecay, batch_size, eval_batch_size, alpha, beta, bptt, optimizer_name, nonmono, save, when, log_interval, clip, criterion):
    # Loop over epochs.
    params = list(model.parameters()) + list(criterion.parameters())
    best_val_loss = []
    stored_loss = 100000000

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        optimizer = None
        # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
        if optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr, weight_decay=wdecay)
        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wdecay)
        for epoch in range(1, epochs+1):
            epoch_start_time = time.time()
            train(corpus, train_data, model_type, model, optimizer, alpha, beta, batch_size, clip, log_interval, epoch, bptt, criterion)
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(corpus, model_type, model, val_data, bptt, criterion, eval_batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                        epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(model, optimizer, criterion, save)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(corpus, model_type, model, val_data, bptt, criterion, eval_batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                  epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(model, optimizer, criterion, save)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                if optimizer_name == 'sgd' and 't0' not in optimizer.param_groups[0] and (len(best_val_loss)>nonmono and val_loss > min(best_val_loss[:nonmono])):
                    print('Switching to ASGD')
                    optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0., weight_decay=wdecay)

                if epoch in when:
                    print('Saving model before learning rate decreased')
                    model_save(model, optimizer, criterion, '{}.e{}'.format(save, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    return model


# Load the best saved model.
def do_evaluation(corpus, save, model_type, test_data, test_batch_size, bptt, criterion):
    model, criterion, optimizer = model_load(save)

    # Run on test data.
    test_loss = evaluate(corpus, model_type, model, test_data, bptt, criterion, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)


def main():
    """Entry point."""
    args = parser.parse_args()
    args.tied = True

    # Set the random seed manually for reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    corpus, train_data, val_data, test_data, batch_size, eval_batch_size, test_batch_size = load_dataset(args.data, args.batch_size)

    model, criterion, optimizer = build_model(corpus,
                                              args.model,
                                              args.emsize,
                                              args.nhid,
                                              args.nlayers,
                                              args.dropout,
                                              args.dropouth,
                                              args.dropouti,
                                              args.dropoute,
                                              args.wdrop,
                                              args.lr,
                                              args.tied,
                                              args.resume,
                                              args.cuda)

    if not args.load_only:
        training_loop(corpus,
                      train_data,
                      val_data,
                      args.model,
                      model,
                      args.lr,
                      args.epochs,
                      args.wdecay,
                      batch_size,
                      eval_batch_size,
                      args.alpha,
                      args.beta,
                      args.bptt,
                      args.optimizer,
                      args.nonmono,
                      args.save,
                      args.when,
                      args.log_interval,
                      args.clip,
                      criterion)

    do_evaluation(corpus, args.save, args.model, test_data, test_batch_size, args.bptt, criterion)


if __name__ == "__main__":
    main()

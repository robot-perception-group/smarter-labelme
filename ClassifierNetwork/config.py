#!/usr/bin/env python3

def config(args):
    return {'train_kwargs': { 'batch_size': args.batch_size, 'num_workers': 4, 'shuffle': True },
            'test_kwargs': { 'batch_size': args.test_batch_size, 'num_workers': 1, 'shuffle': True }
            }

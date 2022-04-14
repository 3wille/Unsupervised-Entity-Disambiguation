#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def batch_with_indices(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        lower_border = ndx
        upper_border = min(ndx + n, l)
        yield iterable[lower_border:upper_border], lower_border, upper_border


def flatten(t):
    flat_list = [item for sublist in t for item in sublist]
    return flat_list

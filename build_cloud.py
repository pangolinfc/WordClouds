#!/usr/bin/env python
from __future__ import print_function

from WordClouds import *

import string
import random
import math
import re

def test_weights():
    """Yield a bunch of randomly generated (word, weight) pairs
    
    Note:
        Actually the words are randomly generated; the weights are not.
    """
    num_words = 0.0
    test_letters = string.lowercase
    while True:
        num_words += 1.0
        next_word = ''.join(
            [ random.choice(test_letters) for i in xrange(random.randint(3,12)) ] )
        yield( next_word, math.sqrt(1.0/num_words) )

def count_tokens(s):
    """Return [ (token, count), (token, count), ... ] for string sorted from
    most common to least common"""
    # Lazy import to limit 2.7 dependency
    try:
        from collections import Counter
    except ImportError:
        raise Exception("count_tokens requires collections.Counter (python 2.7)")

    tokenizer = re.compile(r"(?u)\w[\w'\-_]+\w")
    return Counter(tokenizer.findall(s)).most_common()

if __name__ == '__main__':
    # We have to import pyplot (and thus WordClouds) after we figure out which
    # back-end to use.  So, we have to do all the imports in a funny order 
    # rather than at the top of the module.
    import argparse
    import sys
    import os.path

    description="""Generate a wordcloud image from a list of words and weights.

    Each line of input is a weight and a token, separated by whitespace.
    These do not have to be in sorted order.  With the --count-tokens
    flag you can also use a simple text file; it will then generate its
    own token counts."""

    parser = argparse.ArgumentParser(
            description=description )
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'),
            default=sys.stdin, help="Word/weights file (- for stdin)")
    parser.add_argument('outfile', nargs='?', type=argparse.FileType('wb'),
            default=sys.stdout, help="Output image (- for stdout)")
    parser.add_argument('--bg', type=str,
            help="Background color (eg #FFFFFF), or omit for transparent")
    parser.add_argument('--width', default=None, type=int,
            help="Inches (for pdf, ps, svg) or pixels (for png, X11)")
    parser.add_argument('--height', default=None, type=int,
            help="Inches (for pdf, ps, svg) or pixels (for png, X11)")
    parser.add_argument('--dpi', default=96.0, type=float,
            help="DPI; only meaningful for vector formats")
    parser.add_argument('--time-limit', default=None, type=float, dest='time_limit',
            help="Spend at most TIME_LIMIT seconds generating the cloud" )
    parser.add_argument('--colors', default=None, type=str,
            help="Comma separated list of colors (eg '#FA0000,#00FA00,#0000FA').  The first color will be used for low weight words, the last color for high weight words, and it will interpolate colors for weights in between.")
    parser.add_argument('--format', default=None, type=str,
            help="Format to output to (SVG, PDF, PS, PNG, X11)" )
    parser.add_argument('--count-tokens', default=False, action="store_true",
            dest="count_tokens", help="Input file should be tokenized and counted" )
    parser.add_argument('--seed', default=None, type=int,
            help="The random seed to use when laying out the word cloud" )
    parser.add_argument('--x11', default=False, action="store_true",
            help="Display to the screen instead of saving to a file" )

    parser.add_argument('--split-limit', default=2**-3, type=float, dest='split_limit',
            help="Size of rectangles to approximate words by")
    parser.add_argument('--visual-limit', default=2**-7, type=float, dest='visual_limit',
            help="Smallest word to display")

    args = parser.parse_args()

    if args.x11:
        args.format = 'x11'

    if args.count_tokens:
        token_counts = count_tokens(args.infile.read())
    else:
        # Input is "weight word" (possibly unsorted) but we need (word,weight)
        # (definitely sorted)
        token_counts = list()
        for line in args.infile:
            w,t = line.split(None, 1)
            try:
                fw = float(w)
                if fw <= 0:
                    raise ValueError()

                token_counts.append((t.strip(), fw))
            except ValueError:
                raise ValueError("Invalid weight at line {}".format(len(token_counts)+1))
        token_counts.sort(key=lambda _: _[1], reverse=True)

    show_kwds = dict()
    if args.time_limit is not None:
        show_kwds['time_limit'] = args.time_limit

    save_kwds = dict()
    if args.bg is None:
        save_kwds['transparent'] = True
        args.bg = "#FFFFFF"
        transparent = True
    else:
        save_kwds['facecolor'] = args.bg
        transparent = False

    import matplotlib
    if args.format is None:
        # We attempt to infer the format from the filename...
        args.format = os.path.splitext(args.outfile.name)[1].lstrip('.')

    # We want to set default sizes differently depending on if the format
    # is rasterized (so we are going by pixels) or if the format is
    # vector-based (so we are going by inches)
    raster_formats = [ 'png', 'x11' ]
    vector_formats = [ 'svg', 'pdf', 'ps' ]

    if args.format.lower() in vector_formats:
        if args.width is None:
            figwidth = 4
        else:
            figwidth = args.width

        if args.height is None:
            figheight = 4
        else:
            figheight = args.height
    else:
        if args.width is None:
            figwidth = 512/args.dpi
        else:
            figwidth = args.width/args.dpi

        if args.height is None:
            figheight = 512/args.dpi
        else:
            figheight = args.height/args.dpi

    # Configure matplotlib backends
    if args.format.lower() == 'png':
        matplotlib.use('Agg')
        interactive = False
        pad = 0.0
    elif args.format.lower() in [ 'svg', 'pdf', 'ps' ]:
        matplotlib.use(args.format)
        interactive = False
        pad = 0.0
    elif args.format.lower() == 'x11':
        # Use the default matplotlib backend.  Of course, the default backend
        # might be non-interactive.  I have no idea what will happen in that
        # case.
        interactive = True
        pad = 0.05
    else:
        print("Unrecognized format {}; using png".format(args.format.lower()),
                file=sys.stderr)
        matplotlib.use('Agg')
        interactive = False
        pad = 0.0

    from matplotlib.colors import LinearSegmentedColormap

    from WordClouds import WordCloud

    # Generate the colormap
    if args.colors is not None:
        colors = [ c.strip() for c in args.colors.split(",") ]
        show_kwds['cmap'] = LinearSegmentedColormap.from_list("custom", colors)
    else:
        show_kwds['cmap'] = "copper_r"

    build_params = {
            'split_limit': args.split_limit,
            'visual_limit': args.visual_limit
        }

    wc = WordCloud(token_counts, seed=args.seed, **build_params)

    from matplotlib import pyplot

    fig = pyplot.figure(facecolor=args.bg)
    fig.set_size_inches(figwidth, figheight)
    ax = fig.add_subplot(111)

    # Make the cloud render on-screen as it is generated
    if interactive:
        pyplot.ion()
        pyplot.show()

    wc.show(axes=ax, **show_kwds)
    pyplot.subplots_adjust(pad, pad, 1.0-pad, 1.0-pad, 0, 0)

    if interactive:
        pyplot.ioff()
        pyplot.show()
    else:
        fig.savefig(args.outfile, dpi=args.dpi, **save_kwds)


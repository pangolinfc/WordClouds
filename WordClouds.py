#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
A library to build and show word clouds.  It uses matplotlib for:
    * Display
    * Converting text to paths
    * A convenient rectangle class

Example:

    wordweights = [ 
        ('hello', 5), ('world', 2), ('helloooo', 0.2), ('hi', 0.1) ]
    cloud = WordCloud(wordweights)
    cloud.show()

Or for more fnu:

    from matplotlib import pyplot

    pyplot.ion()

    cloud.show()

    pyplot.ioff()
    pyplot.show()

Here's the algorithm:
    Place the first word at the origin
    While words remain:
        If the word is big:
            If the cloud is wider than it is tall:
                Place the word randomly along the top or bottom edges [1]
            If the cloud is taller than it is wide:
                Place the word randomly along the left or right edges [1]
        If the word is small:
            Place the word randomly along any edge

        Alternate pushing[2] the word toward the x- and y-axes until it won't
        move any more

Obviously there's a lot more going on, but that's the basic idea.

[1] The distribution has been chosen carefully; see random_position()
[2] This `pushing' accounts for a large percentage of the code.  What you're
    really doing is moving the word all the way to the axis, then moving it
    back toward its original position until you find the first place it fits.

Other things that are going on:
    *   Once a word has been inserted it is approximated by rectangles.  These
        are then inserted into an index.  
    *   New words have their entire bounding box compared against the 
        rectangles already in the index.  They are not broken up into smaller
        rectangles until they are already in the cloud.
"""

# Standard libraries
from __future__ import print_function
from itertools import *
import sys
import random
import time
import math

from pprint import pprint

# External libraries
import matplotlib
from matplotlib import pyplot, patches, textpath, transforms, cm, font_manager
from matplotlib.patches import PathPatch, FancyBboxPatch
from matplotlib.transforms import Affine2D
from matplotlib.path import Path
# Note that TextPath requires matplotlib >= 1.1
from matplotlib.textpath import TextPath

from quadtree import Quadtree

# Local libraries
import BoxifyWord

__all__ = [ 'build_cloud', 'WordCloud' ]

class BboxQuadtree(Quadtree):
    """
    This is just a convenience wrapper for Quadtree to make it accept Bboxes
    from matplotlib instead of just tuples.

    Quadtree is a wrapper for a C quadtree library which indexes tuples of
    the form:
        ( left, bottom, right, top )
    So to insert:
        quadtree.add_bbox(i, bboxes[i])

    When you want to query the quadtree you just do this:

        box_indices = quadtree.bbox_intersection(bbox)
        bboxes = [ bboxes[i] for i in box_indices ]

    Notice that you have to maintain your own lookup table of Bboxes.

    FIXME: There's no reason this class can't maintain its own LUT of
        Bboxes instead of requiring the user to maintain an external one.
        The original Quadtree class was written for a far more general purpose
        than I need.
    """
    def __init__(self, *args, **kwds):
        super(BboxQuadtree, self).__init__(*args, **kwds)
        self.coverage = [ 0.0, 0.0, 0.0, 0.0 ]

    @staticmethod
    def coverage_areas(bbox):
        E = min(bbox.xmax, 0.0) - min( bbox.xmin, 0.0 )
        W = max(bbox.xmax, 0.0) - max( bbox.xmin, 0.0 )
        S = min(bbox.ymax, 0.0) - min( bbox.ymin, 0.0 )
        N = max(bbox.ymax, 0.0) - max( bbox.ymin, 0.0 )
        return [ N*E, N*W, S*E, S*W ]

    def update_coverage(self, bbox):
        self.coverage = [ orig+new 
                for orig, new in zip(self.coverage, self.coverage_areas(bbox)) ]

    def add_bbox(self, i, bbox):
        """Insert bbox into the Quadtree index.

        Note that if you ever want to retrieve the Bboxes you insert you'll
        need to keep your own external table.  The parameter `i' should be
        an index into that table.

        Args:
            i: An index into your own external list of bboxes
            bbox: The Bbox to insert
        """

        return self.add( i, ( bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax ) )

    def bbox_intersection(self, bbox):
        """Query the quadtree for Bboxes intersecting bbox.

        Args:
            bbox: A matplotlib.transforms.Bbox region to query.

        Returns:
            Indexes of Bboxes (possibly) intersecting bbox.  You'll want to
            double check and make sure there were no false positives.
        """
        return self.likely_intersection( (bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax) )

def find_space(size, intervals, pad=0.0, maximum=None):
    """Find space for an interval of width size in a list of intervals

    Search through a list of closed intervals, starting at the lowest value in
    the list, until a space of width at least size*(1+2*pad) is found.  Then
    return the position at which the interval should be inserted in order
    to have at least size*pad space on each side.

    CAVEAT: This function assumes that the space between -0.5*size and the
    lowest interval is empty (if the lowest interval doesn't overlap -0.5*size)

    Args:
        size: The size of interval to squeeze in
        intervals: A list of tuples of the form:
                    [ (a0, b0), (a1, b1), ... ]
                representing closed intervals.  They should be sorted so that
                    a0 < a1, a1 < a2, ... 
                and such that their upper bounds are all positive.
        pad: Pad size by a factor of pad on each side before finding a space 
                for it.
        maximum: If the lower bound gets higher than maximum, return maximum
                instead.  `None' means no maximum.

    Returns:
        A number lower_bound representing where the bottom of the interval
        should be inserted.  In other words, if the interval 
            (lower_bound, lower_bound+size)
        were inserted into intervals it would not overlap with any of them.
    """
    if len(intervals) == 0:
        return -0.5*size
    lower_bound = -0.5*size # 0 

    size_pad = pad*size
    size = size*(1.0+2*pad)
    for interval in intervals: # i in xrange(len(intervals)):
        if interval[0] <= lower_bound:
            lower_bound = max(lower_bound, interval[1]) + size_pad
        else:
            if interval[0] - lower_bound > size:
                return lower_bound
            else:
                lower_bound = interval[1] + size_pad

        if maximum is not None and lower_bound > maximum:
            return maximum

    return lower_bound

def kisses_y(bbox, other):
    """Returns True if two boxes are kissing along a vertical edge.

    `Kissing' means they overlap only on their edge.
    """
    return (    (bbox.ymax == other.ymin and bbox.ymin <= other.ymin)
            or  (bbox.ymin == other.ymax and bbox.ymin >= other.ymin) )

def kisses_x(bbox, other):
    """Returns True if two boxes are kissing along a horizontal edge.

    `Kissing' means they overlap only on their edge.
    """
    return (    (bbox.xmax == other.xmin and bbox.xmin <= other.xmin)
            or  (bbox.xmin == other.xmax and bbox.xmin >= other.xmin) )

def push_bbox_down(bbox, bboxes, index):
    """Attempt to slide bbox as close to the x-axis as possible, starting
    from above.

    Really this function does the following:
        1. Suppose bbox is sitting in a field of other bboxes.
        2. Slide the bbox down to the x-axis
        3. Move the bbox upward toward its original position until it no 
            longer overlaps another box
        4. If no space can be found, place it back in its original position.

    Args:
        bbox: The matplotlib.transforms.Bbox to move
        bboxes: A list of Bboxes to pack around
        index: A Quadtree index with the same contents as bboxes

    Returns:
        A tuple (x, y) that should be the new center for bbox.
    """
    size = abs(bbox.height)
    left = bbox.xmin
    right = bbox.xmax
    bottom = min(-0.5*size, bbox.ymin)
    top = bbox.ymax

    search_bbox = matplotlib.transforms.Bbox(((left, bottom), (right, top)))
    
    intervals = list()
    for i in index.bbox_intersection( search_bbox ):
        if search_bbox.overlaps(bboxes[i]) and not kisses_x(bbox, bboxes[i]):
            intervals.append( (bboxes[i].ymin, bboxes[i].ymax) )
    intervals.sort(key=lambda t: t[0])

    lower_bound = find_space(size, intervals, maximum=bbox.ymin)

    xpos = 0.5*(right+left)
    ypos = lower_bound + 0.5*size

    return (xpos, ypos)

def push_bbox_left(bbox, bboxes, index):
    size = abs(bbox.width)
    left = min(-0.5*size, bbox.xmin)
    right = bbox.xmax
    bottom = bbox.ymin
    top = bbox.ymax

    search_bbox = matplotlib.transforms.Bbox(((left, bottom), (right, top)))
    
    intervals = list()
    for i in index.bbox_intersection( search_bbox ):
        if search_bbox.overlaps(bboxes[i]) and not kisses_y(bbox, bboxes[i]):
            intervals.append( (bboxes[i].xmin, bboxes[i].xmax) )
    intervals.sort(key=lambda t: t[0])

    lower_bound = find_space(size, intervals, maximum=bbox.xmin)

    xpos = lower_bound + 0.5*size
    ypos = 0.5*(top+bottom)

    return (xpos, ypos)

def push_bbox_up(bbox, bboxes, index):
    size = abs(bbox.height)
    left = bbox.xmin
    right = bbox.xmax
    bottom = bbox.ymin
    top = max(0.5*size, bbox.ymax)

    search_bbox = matplotlib.transforms.Bbox(((left, bottom), (right, top)))
    
    intervals = list()
    for i in index.bbox_intersection( search_bbox ):
        if search_bbox.overlaps(bboxes[i]) and not kisses_x(bbox, bboxes[i]):
            intervals.append( (-bboxes[i].ymax, -bboxes[i].ymin) )
    intervals.sort(key=lambda t: t[0])

    lower_bound = -find_space(size, intervals, maximum=-bbox.ymax)

    xpos = 0.5*(right+left)
    ypos = lower_bound - 0.5*size

    return (xpos, ypos)

def push_bbox_right(bbox, bboxes, index):
    size = abs(bbox.width)
    left = bbox.xmin
    right = max(0.5*size, bbox.xmax)
    bottom = bbox.ymin
    top = bbox.ymax

    search_bbox = matplotlib.transforms.Bbox(((left, bottom), (right, top)))
    
    intervals = list()
    for i in index.bbox_intersection( search_bbox ):
        if search_bbox.overlaps(bboxes[i]) and not kisses_y(bbox, bboxes[i]):
            intervals.append( (-bboxes[i].xmax, -bboxes[i].xmin) )
    intervals.sort(key=lambda t: t[0])

    lower_bound = -find_space(size, intervals, maximum=-bbox.xmax)

    xpos = lower_bound - 0.5*size
    ypos = 0.5*(top+bottom)

    return (xpos, ypos)

def random_position(a, b):
    """Draw from a bimodal distribution with support in (a,b).

    This is just two triangular distributions glued together.  Empirically
    it's a nice way to build the word clouds, but it should be replaced at
    some point with something more sophisticated.  See the note below.

    Here's the idea behind the crazy bimodal distribution:
        *   If we drop the words too close to the axes they'll just stack up
            and we get something that looks like a plus
        *   If we drop a word at the corner of the region it'll stack right
            above the previous word dropped from that direction, then get
            shoved left.  The result looks like a square.
        *   Dropping halfway between the axis and the corner will result in
            something halfway in-between (usually a hexagon)

    I think that what this method is `approximating' is the following:
        *   Divide the area into quadrants
        *   Keep track of the area covered in each quadrant
        *   Drop the new word in a random position on the edge of the
            quadrant with the least coverage
    """
    if random.choice([ 0, 1 ]):
        return random.triangular(a, 0.5*(a+b))
    else:
        return random.triangular(0.5*(a+b), b)

def build_cloud(wordweights, 
        loose=False, seed=None, split_limit=2**-3, pad=1.10, visual_limit=2**-5,
        highest_weight=None ):
    """Convert a list of words and weights into a list of paths and weights.

    You should only use this function if you know what you're doing, or if
    you really don't want to cache the generated paths.  Otherwise just use
    the WordCloud class.

    Args:
        wordweights: An iterator of the form 
                [ (word, weight), (word, weight), ... ]
            such that the weights are in decreasing order.
        loose: If `true', words won't be broken up into rectangles after
            insertion.  This results in a looser cloud, generated faster.
        seed: A random seed to use
        split_limit: When words are approximated by rectangles, the rectangles
            will have dimensions less than split_limit.  Higher values result
            in a tighter cloud, at a cost of more CPU time.  The largest word
            has height 1.0.
        pad: Expand a word's bounding box by a factor of `pad' before
            inserting it.  This can actually result in a tighter cloud if you
            have many small words by leaving space between large words.
        visual_limit: Words with height smaller than visual_limit will be
            discarded.
        highest_weight: Experimental feature.  If you provide an upper bound
            on the weights that will be seen you don't have to provide words
            and weights sorted.  The resulting word cloud will be noticeably
            uglier.

    Generates:
        Tuples of the form (path, weight) such that:
            * No two paths intersect
            * Paths are fairly densely packed around the origin
            * All weights are normalized to fall in the interval [0, 1]
    """
    if seed is not None:
        random.seed(seed)

    font_properties = font_manager.FontProperties(
                family="sans", weight="bold", stretch="condensed")
    xheight = TextPath((0,0), "x", prop=font_properties).get_extents().expanded(pad,pad).height

    # These are magic numbers.  Most wordclouds will not exceed these bounds.
    # If they do, it will have to re-index all of the bounding boxes.
    index_bounds = (-16, -16, 16, 16)
    index = BboxQuadtree(index_bounds)

    if highest_weight is None:
        # Attempt to pull the first word and weight.  If we fail, the wordweights
        # list is empty and we should just quit.
        #
        # All this nonsense is to ensure it accepts an iterator of words
        # correctly.
        iterwords = iter(wordweights)
        try:
            first_word, first_weight = iterwords.next()
            iterwords = chain([(first_word, first_weight)], iterwords)
        except StopIteration:
            return

        # We'll scale all of the weights down by this much.
        weight_scale = 1.0/first_weight
    else:
        weight_scale = 1.0/highest_weight
        iterwords = iter(wordweights)

    bboxes = list()

    bounds = transforms.Bbox(((-0.5, -0.5), (-0.5, -0.5)))
    for tword, tweight in iterwords:
        weight = tweight*weight_scale
        if weight < visual_limit:
            # You're not going to be able to see the word anyway.  Quit
            # rendering words now.
            continue

        word_path = TextPath((0,0), tword, prop=font_properties)
        word_bbox = word_path.get_extents().expanded(pad, pad)
        # word_scale = weight/float(word_bbox.height)
        word_scale = weight/float(xheight)
        
        # When we build a TextPath at (0,0) it doesn't necessarily have
        # its corner at (0,0).  So we have to translate to the origin,
        # scale down, then translate to center it.  Feel free to simplify
        # this if you want.
        word_trans = Affine2D.identity().translate(
                                -word_bbox.xmin,
                                -word_bbox.ymin
                            ).scale(word_scale).translate(
                                -0.5*abs(word_bbox.width)*word_scale,
                                -0.5*abs(word_bbox.height)*word_scale )

        word_path = word_path.transformed(word_trans)

        word_bbox = word_path.get_extents().expanded(pad, pad)

        if weight > split_limit:
            # Big words we place carefully, trying to make the dimensions of
            # the cloud equal and center it around the origin.
            gaps = ( 
                    ("left", bounds.xmin), ("bottom", bounds.ymin), 
                    ("right", bounds.xmax), ("top", bounds.ymax) )
            direction = min(gaps, key=lambda g: abs(g[1]))[0]
        else:
            # Small words we place randomly.
            direction = random.choice( [ "left", "bottom", "right", "top" ] )

        # Randomly place the word along an edge...
        if direction in ( "top", "bottom" ):
            center = random_position(bounds.xmin, bounds.xmax)
        elif direction in ( "right", "left" ):
            center = random_position(bounds.ymin, bounds.ymax)

        # And push it toward an axis.
        if direction == "top":
            bbox = word_bbox.translated( center, index_bounds[3] )
            xpos, ypos = push_bbox_down( bbox, bboxes, index )
        elif direction == "right":
            bbox = word_bbox.translated( index_bounds[2], center )
            xpos, ypos = push_bbox_left( bbox, bboxes, index )
        elif direction == "bottom":
            bbox = word_bbox.translated( center, index_bounds[1] )
            xpos, ypos = push_bbox_up( bbox, bboxes, index )
        elif direction == "left":
            bbox = word_bbox.translated( index_bounds[0], center )
            xpos, ypos = push_bbox_right( bbox, bboxes, index )
    
        # Now alternate pushing the word toward different axes until either
        # it stops movign or we get sick of it.
        max_moves = 2
        moves = 0
        while moves < max_moves and (moves == 0 or prev_xpos != xpos or prev_ypos != ypos):
            moves += 1
            prev_xpos = xpos
            prev_ypos = ypos
            if direction in ["top", "bottom", "vertical"]:
                if xpos > 0:
                    bbox = word_bbox.translated( xpos, ypos )
                    xpos, ypos = push_bbox_left( bbox, bboxes, index )
                elif xpos < 0:
                    bbox = word_bbox.translated( xpos, ypos )
                    xpos, ypos = push_bbox_right( bbox, bboxes, index )
                direction = "horizontal"
            elif direction in ["left", "right", "horizontal"]:
                if ypos > 0:
                    bbox = word_bbox.translated( xpos, ypos )
                    xpos, ypos = push_bbox_down( bbox, bboxes, index )
                elif ypos < 0:
                    bbox = word_bbox.translated( xpos, ypos )
                    xpos, ypos = push_bbox_up( bbox, bboxes, index )
                direction = "vertical"

        wordtrans = Affine2D.identity().translate( xpos, ypos )

        transpath = word_path.transformed(wordtrans)
        bbox = transpath.get_extents()

        # Swallow the new word into the bounding box for the word cloud.
        bounds = matplotlib.transforms.Bbox.union( [ bounds, bbox ] )

        # We need to check if we've expanded past the bounds of our quad tree.
        # If so we'll need to expand the bounds and then re-index.
        new_bounds = index_bounds
        while not BoxifyWord.bbox_covers(
            # FIXME: Why am I not just doing this with a couple of logarithms?
                    matplotlib.transforms.Bbox(((new_bounds[0], new_bounds[1]),
                                                (new_bounds[2], new_bounds[3]))),
                    bounds ):
            new_bounds = tuple( map( lambda x: 2*x, index_bounds ) )

        if new_bounds != index_bounds:
            # We need to re-index.
            index_bounds = new_bounds
            index = BboxQuadtree(index_bounds)
            for i, b in enumerate(bboxes):
                index.add_bbox(i, b)

        # Approximate the new word by rectangles (unless it's too small) and
        # insert them into the index.
        if not loose and max(abs(bbox.width), abs(bbox.height)) > split_limit:
            for littlebox in BoxifyWord.splitword( 
                    bbox, transpath, limit=split_limit ):
                bboxes.append( littlebox )
                index.add_bbox( len(bboxes)-1, littlebox )
        else:
            bboxes.append( bbox )
            index.add_bbox( len(bboxes)-1, bbox )

        yield (transpath, weight)

class WordCloud:
    """A WordCloud represents a single word cloud, ready to be displayed on
    your screen.

    The WordCloud accepts a list of tuples of the form
        [ ( word_0, weight_0 ), ( word_1, weight_1 ), ... ]
    such that word_i >= word_i+1.  It then generates a word cloud suitable
    for display in matplotlib.

    WordCloud generates the word cloud lazily.  In other words:
        * Instantiating the WordCloud will be very fast
        * The cloud will build during the first render
        * Subsequent renders will be very fast
    If you want to avoid this behavior specify `lazy=False' when
    instantiating the WordCloud.
    """
    def __init__(self, word_weights, 
            lazy=True, label=None, cmap=None, **kwds):
        """Build a WordCloud.

        Args:
            word_weights: An iterator containing tuples of the form:
                    [ ( word, weight ), ( word, weight ), ... ]
            lazy: Build the word cloud right away, not lazily.
            label: A label to draw on the cloud
            cmap: A name of a matplotlib colormap to use (or a cmap)

            All other args are passed to build_cloud; the one of most
            interest is likely to be:

            loose: Speed up the draw by only checking for bounding box
                collisions, not word collisions.
        """
        self.build_params = {
                'loose': False,
                'seed': None,
                'split_limit': 2**-3,
                'visual_limit': 2**-5,
                'pad': 1.10 }
        self.build_params.update(kwds)

        self.label = label
        if cmap is None or isinstance(cmap, basestring):
            self.cmap = cm.get_cmap(cmap)
        else:
            self.cmap = cmap

        if lazy:
            self.word_path_iter = build_cloud( word_weights, **self.build_params )
            self.word_paths = list()
        else:
            self.word_path_iter = None
            self.word_paths = list(build_cloud( word_weights, **self.build_params ))

    def __iter__(self):
        # The first iter() will yield the paths from build_cloud() as they 
        # are built.  It will cache these in a list, which it yields in future 
        # iter()s.

        # It knows which to return because next() will set word_path_iter to
        # None once it's exhausted.
        if self.word_path_iter is not None:
            return self
        else:
            return iter(self.word_paths)

    def next(self):
        try:
            # If there's anything left in word_path_iter, cache and return
            # it...
            next_path, next_weight = self.word_path_iter.next()
            self.word_paths.append( (next_path, next_weight) )
            return (next_path, next_weight)
        except StopIteration:
            # Otherwise set word_path_iter to None so iter() will know its
            # exhausted and raise StopIteration again.
            self.word_path_iter = None
            raise

    def show(self, axes=None, step=0.25, cmap=None, label=None, interact=None, 
            background_color="white", time_limit=None):
        """Render the wordcloud to the screen or provided set of axes.
        
        This function can take advantage of pylab's interactive mode, so:
            pyplot.ion()

            cloud.render(axes)

            pyplot.ioff()

        will update the word cloud on the screen as it is computed.

        Args:
            axes: A set of matplotlib axes to draw to (or None)
            step: In interactive mode, re-draw the screen every step seconds.
            cmap: The name of a matplotlib colormap or a matplotlib colormap
            label: A text label to draw on the plot
            interact: Interactive updates (if pyplot.isinteractive() == True)
            background_color: The background color for the word cloud
        """
        start_time = time.clock()

        if interact is None:
            interact = ( not self.is_built() )

        interact = (interact and pyplot.isinteractive())

        if axes is None:
            show_plot = True
            axes = pyplot.figure().gca()
            # axes.set_axis_bgcolor("white")
            axes.get_figure().set_facecolor(background_color)
        else:
            show_plot = False

        if label is None:
            label = self.label

        if cmap is None:
            cmap = self.cmap
        elif isinstance(cmap, basestring):
            cmap = cm.get_cmap(cmap)

        last_draw = time.time()

        axes.set_animated(True)
        axes.set_aspect("equal")
        axes.set_axis_off()
        if label is not None:
            axes.set_title( label, verticalalignment="bottom",
                    bbox=dict(facecolor="white", boxstyle="round,pad=0.2") )
        
        if interact:
            pyplot.draw()

        bounds = matplotlib.transforms.Bbox(((0,0), (1,1)))
        axes.set_xlim(bounds.xmin, bounds.xmax)
        axes.set_ylim(bounds.ymin, bounds.ymax)
        for path, weight in self:

            box = path.get_extents()
            axes.add_patch( PathPatch( path, lw=0, facecolor=cmap(weight) ) )

            bounds = matplotlib.transforms.Bbox.union( [ bounds, box ] )

            axes.set_xlim(bounds.xmin, bounds.xmax)
            axes.set_ylim(bounds.ymin, bounds.ymax)
            
            if time_limit is not None and time.clock() > start_time + time_limit:
                break
            elif interact and time.time() > last_draw+step:
                pyplot.draw()
                last_draw = time.time()

        if interact:
            pyplot.draw()

        axes.set_animated(False)
        
        if show_plot:
            pyplot.show()

    def save(self, filename, **kwds):
        """Write the wordcloud to a file.

        A wrapper for Figure.savefig() to save a word cloud as an image.  It
        determines the format from the filename extension.

        Unlike Figure.savefig(), this function renders a transparent background
        by default.

        Args:
            filename: The name of the image file to save to

            All other keyword arguments are passed to Figure.savefig().
        """

        if 'transparent' not in kwds:
            kwds['transparent'] = True

        fig = pyplot.figure()
        self.show(fig.gca())
        fig.savefig( filename, **kwds )
        pyplot.close()

    def is_built(self):
        """Return True if the wordcloud has been built

        This function is required if you want to pickle a wordcloud.  Since
        pickle can't handle generators you need to know if the WordCloud 
        actually exists as a collection of paths, or if it is merely a gener-
        ator; if it's a generator consider rebuilding it with lazy=False.
        """
        return self.word_path_iter is None

import unittest

class WordCloudTestCase(unittest.TestCase):
    def setUp(self):
        words = [ "alpha", "beta" ] * 50
        weights = map( lambda w: 1.0/(w+1), range(100) )
        self.word_weights = zip(words, weights)
        self.word_cloud = WordCloud(self.word_weights, seed=1)

    def test_build(self):
        """Make sure a simple cloud builds without errors"""
        for path, weight in self.word_cloud:
            pass

    def test_show(self):
        """Make sure a simple cloud shows without errors"""
        ax = pyplot.gca()
        pyplot.ioff()
        self.word_cloud.show(ax)
        pyplot.close()

    def test_interactive_show(self):
        """Make sure a simple cloud shows non-interactively without errors"""
        ax = pyplot.gca()
        pyplot.ion()
        self.word_cloud.show(ax)
        pyplot.close()
    
    def test_caching(self):
        """Make sure the cloud is cached correctly after the first run"""
        self.assertTrue(
                self.word_cloud.word_path_iter is not None, 
                "Word cloud should not be cached yet" )
        run1 = list(iter(self.word_cloud))
        self.assertTrue(
                self.word_cloud.word_path_iter is None, 
                "Word cloud should be cached after first run" )
        run2 = list(iter(self.word_cloud))
        self.assertEqual(run1, run2,
                "Subsequent runs don't return the same cloud" )

    def test_iter(self):
        from numpy import allclose

        """Make sure build_cloud accepts an iterator of words correctly"""
        run1 = list(iter(self.word_cloud))
        run2 = list(iter(WordCloud(iter(self.word_weights), seed=1)))

        self.assertEqual(len(run1), len(run2), "The WordClouds were different sizes")

        for (p1, w1), (p2, w2) in zip(run1, run2):
            if not allclose(p1.vertices, p2.vertices) or w1 != w2:
                self.assertEqual(run1, run2, 
                        "Failed to properly process iterator input")
                return

    def test_collisions(self):
        """Ensure that no pair of paths intersects."""

        # collided_paths = list()
        
        paths = list()
        for p, w in self.word_cloud:
            paths.extend(BoxifyWord.cleaned_textpath(p))

        for p1, p2 in combinations(paths, 2):
            if( p1.get_extents().overlaps(p2.get_extents())
                    and p1.intersects_path(p2, filled=True) ):
                # collided_paths.extend((p1, p2))
                self.fail("Found a collision between two paths with holes")

        # ax = pyplot.figure().gca()
        # pyplot.ion()
        # self.word_cloud.show(ax)
        # for p in collided_paths:
        #     ax.add_patch( PathPatch( p, lw=1, facecolor='blue', alpha=0.2 ) )
        # pyplot.ioff()
        # pyplot.show()

    def test_collisions_noloops(self):
        """Ensure that no pair of paths without holes intersects."""

        words = [ "sun", "luck" ] * 50
        weights = map( lambda w: 1.0/(w+1), range(100) )
        word_weights = zip(words, weights)
        word_cloud = WordCloud(word_weights, seed=1)
        
        paths = list()
        for p, w in word_cloud:
            paths.extend(BoxifyWord.cleaned_textpath(p))

        for p1, p2 in combinations(paths, 2):
            if( p1.get_extents().overlaps(p2.get_extents())
                    and p1.intersects_path(p2, filled=True) ):
                self.fail("Found a collision between two paths without holes")


if __name__ == '__main__':

    """
    import re
    from collections import defaultdict
    import math

    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        test_file = sys.argv[0]

    token_counts = defaultdict(lambda : 0)
    with open(test_file, "r") as test_handle:
        for line in test_handle:
            for token in re.findall("\w{3,}", line):
                token_counts[token] += 1

    # token_weights = ( (t, math.sqrt(c)) for (t,c) in token_counts.iteritems() )
    token_weights = token_counts.iteritems()

    test_pairs = sorted(token_weights, key=lambda p: p[1], reverse=True)
    test_weights = [ (t, math.sqrt(c)) for t, c in test_pairs ]
    cloud = WordCloud(test_weights, split_limit=2**-3, visual_limit=0, seed=1)

    # test_weights = [ (t, math.sqrt(c)) for t, c in token_weights ]
    # highest_weight = max(test_weights, key=lambda tw: tw[1])[1]
    # cloud = WordCloud(test_weights, highest_weight=highest_weight)

    pyplot.ion()
    
    cloud.show(cmap="jet", label=test_file)

    pyplot.ioff()
    pyplot.show()
    """

    """
    import string
    import time

    def test_weights():
        num_words = 0.0
        test_letters = string.lowercase
        while True:
            num_words += 1.0
            next_word = ''.join(
                [ random.choice(test_letters) for i in xrange(random.randint(3,12)) ] )
            yield( next_word, math.sqrt(1.0/num_words) )

    pyplot.ion()

    for duration in [ 1.0, 5.0, 20.0, 60.0 ]:

        wc = WordCloud(test_weights(), visual_limit=2**-6)

        start_time = time.time()
        end_time = start_time
        paths_processed = 0
        for p, w in wc:
            paths_processed += 1
            if time.time() - start_time > duration:
                end_time = time.time()
                break

        print(paths_processed, "paths processed in", end_time-start_time, "seconds")
        wc.word_path_iter = None
        wc.show()

    pyplot.ioff()
    pyplot.show()
    """


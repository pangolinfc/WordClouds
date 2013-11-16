#!/usr/bin/env python
"""
This small library has two purposes:

    * Approximate a path by a collection of rectangles (in order to speed up
      intersection checking)
    * Work around a bug in matplotlib relating to path intersection checks

The only thing you should need to use is BoxifyWord.splitword(your_text_path).
"""

# Standard libraries
from __future__ import print_function
import sys

# External libraries
import matplotlib
from matplotlib import patches, textpath
from matplotlib.patches import PathPatch, FancyBboxPatch
from matplotlib.path import Path
# TextPath requires matplotlib >= 1.1
from matplotlib.textpath import TextPath

def splitbox(box, tpath, limit=2**-6):
    """Approximate the portion of tpath contained in box by a collection of boxes.

    For a better approximation, consider decreasing limit.

    This works kind of like a binary space partition.  We recursively split
    the root box in half along its wider dimension.  At each level we throw
    away any children that don't intersect the path.

    As we exit the recursion there is a `gluing' phase where we watch to see
    if all of a box's children intersected the path.  If they all did, we 
    return the original box instead of the list of children.

    Args:
        box: A matplotlib.transforms.Bbox known to intersect tpath
        tpath: A matplotlib.path.Path
        limit: All boxes returned will have dimensions exceeding half the
            limit.

    Returns:
        A tuple (full, boxes):
            full: `True' iff every child of this box intersected the path
            boxes: A list of the boxes intersecting the path
    """
    # We assume that box is known to intersect tpath.

    if max(abs(box.width), abs(box.height)) < limit:
        # We're below the limit, so just return the provided box.  The `true'
        # means that all of the child boxes also intersected.
        return (True, [ box ])
    
    # Split the current box in the larger direction...
    if abs(box.width) < abs(box.height):
        boxes = box.splity(0.5)
    else:
        boxes = box.splitx(0.5)

    # Check to see which children intersect the box.  Throw out the ones that
    # don't, and recurse on the ones that do.
    retboxes = []
    full = True
    for subbox in boxes:
        if tpath.intersects_bbox(subbox, filled=True):
            subfull, subboxes = splitbox(subbox, tpath, limit)
            full =  ( full and subfull )
            retboxes.extend(subboxes)
        else:
            # Some child didn't intersect the path.
            full = False

    if full:
        # This box didn't need to be split; all of its children intersect the
        # path.  Instead of returning some huge collection of boxes we just
        # return this one.
        return (True, [ box ])
    else:
        return (False, retboxes)

def splitpath(box, thepath, **kwds):
    """Approximate the given path by rectangles.

    Args:
        box: A matplotlib.transforms.Bbox known to intersect tpath
        thepath: A matplotlib.path.Path
        limit: All boxes returned will have dimensions exceeding half the
            limit.

    Returns:
        boxes: A list of the boxes intersecting the path, all of which lie
            within box.
    """
    return splitbox( box, thepath, **kwds )[1]

def bbox_covers(bbox, other_bbox):
    """Return True if bbox completely covers other_bbox
    """
    return ( bbox.contains(*(other_bbox.max)) and bbox.contains(*(other_bbox.min)) )

def cleaned_textpath(text_path):
    """Prepare a matplotlib.textpath.TextPath for intersection checking

    There's a rather nasty little bug hidden in matplotlib's TextPath code
    which causes path intersections to fail.  This works around it by cleaning
    up (and simplifying) the paths.  It is not perfect, as you would see if
    you were to try displaying the resulting paths.

    The basic problem is that vertices associated with a CLOSEPOLY path code
    are supposed to be ignored.  They are in the case of rendering the path
    (which is what made this bug so horrible to track down) but not when
    you are doing other operations, like path intersection or simplifying.
    Since TextPath (for whatever reason) associates gibberish vertices with
    CLOSEPOLY codes, path intersection checks will behave strangely.

    This code splits compound paths into separate paths, and moves the vertex
    for a CLOSEPOLY to the first vertex in the path.

    This eliminates holes in letters such as `e' or `O' but that's a price we
    have to pay.

    Args:
        text_path: A matplotlib.textpath.TextPath

    Returns:
        A list of cleaned up paths.
    """

    paths = list()
    verts = list()
    codes = list()

    for v,c in text_path.iter_segments(curves=False, simplify=False):
        if c == Path.CLOSEPOLY:
            vert = verts[0]
        else:
            vert = v

        if c == Path.MOVETO and len(verts) > 0:
            # We've started a new path.
            newpath = Path(verts, codes)
            newbox = newpath.get_extents()
            
            if len(paths) == 0:
                # If there are no paths we definitely add this one.\
                paths.append(newpath)
            else:
                lastbox = paths[-1].get_extents()
                if bbox_covers(newbox, lastbox):
                    # If this path covers the last one, replace it.
                    paths[-1] = newpath
                elif not bbox_covers(lastbox, newbox):
                    # If the last path doesn't cover this one, throw it out.
                    paths.append(newpath)

            verts = [vert]
            codes = [c]
        else:
            verts.append(vert)
            codes.append(c)

    # Finally, deal with the last path, which may not be explicitly closed.
    if len(verts) > 0:
        newpath = Path(verts, codes)
        newbox = newpath.get_extents()
        if len(paths) == 0:
            paths.append(newpath)
        else:
            lastbox = paths[-1].get_extents()
            if bbox_covers(newbox, lastbox):
                paths[-1] = newpath
            elif not bbox_covers(lastbox, newbox):
                paths.append(newpath)
    
    return paths

def splitword(box, text_path, **kwds):
    """Approximate the given TextPath by rectangles.

    The difference between this function and splitpath is that this function
    will clean up your TextPath before approximating it, working around
    various matplotlib bugs.

    Args:
        box: A matplotlib.transforms.Bbox known to intersect tpath
        thepath: A matplotlib.path.Path
        limit: All boxes returned will have dimensions exceeding half the
            limit.

    Returns:
        boxes: A list of the boxes intersecting the path, all of which lie
            within box.
    """

    boxes = list()
    for p in cleaned_textpath(text_path):
        boxes.extend(splitpath( p.get_extents(), p, **kwds))

    return boxes

if __name__ == '__main__':

    import numpy
    from pprint import pprint
    from itertools import islice

    import matplotlib.pyplot as pyplot

    axes = pyplot.gca()

    test_word = "Phillip Seymore Hoffman" # "pearly"

    test_path = TextPath( (0,0), test_word )

    top_box = test_path.get_extents()
    boxes = splitword(top_box, test_path, limit=1)

    # axes.add_patch( PathPatch( test_path, lw=1, facecolor="grey" ) )

    
    for p in cleaned_textpath(test_path):
        axes.add_patch( PathPatch( p, lw=1, facecolor='red', alpha=0.2 ) )

    for box in boxes:
        axes.add_patch( 
                FancyBboxPatch( 
                    (box.xmin, box.ymin), 
                    abs(box.width), abs(box.height),
                    boxstyle="square,pad=0.0",
                    facecolor=(0, 0, 1.0), alpha=0.2 ) )

    axbox = top_box.expanded(1.10, 1.10)

    axes.set_xlim(axbox.xmin, axbox.xmax)
    axes.set_ylim(axbox.ymin, axbox.ymax)

    pyplot.show()

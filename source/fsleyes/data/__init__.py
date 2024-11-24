#!/usr/bin/env python
#
# __init__.py - FSLeyes overlay types and data-related utilities.
#
# Author: Paul McCarthy <pauldmccarthy@gmail.com>
#
"""The :mod:`fsleyes.data` module contains FSLeyes overlay data types and
some data-related utilities.

Most FSLeyes overlay data types are defined in the ``fslpy`` library
(e.g. :class:`fsl.data.image.Image`, :class:`fsl.data.mesh.Mesh`). This
sub-package provides some additional overlay types that can be displayed with
FSLeyes:

.. autosummary::
   :nosignatures:

   ~fsleyes.data.tractogram.Tractogram
"""

#==================================== numpy support ====================================#
import os
import re
import numpy as np
import nibabel as nib
from fsleyes.__main__ import PATH
#================================ LayeredArray support =================================#
import _pickle as pickle
script = """
import numpy as np
class LayeredArray:
    def __init__(self, data):
        self.setData(data)
    def setData(self, data):
        pass
    def getData(self):
        ret = np.zeros(self.shape, dtype=self.dtype)
        for i in range(self.shape[3]):
            ret[self.bounds[i][0,0]:self.bounds[i][0,1],
                self.bounds[i][1,0]:self.bounds[i][1,1],
                self.bounds[i][2,0]:self.bounds[i][2,1],i] = self.data[i]
        return ret
    def save(self, path):
        pass
"""
import imp, sys
moduleName = 'LayeredArray'
module = imp.new_module(moduleName)
exec(script, module.__dict__)
sys.modules[moduleName] = module
#=======================================================================================#

import os.path as op

import fsl.utils.path          as fslpath
import fsl.data.utils          as dutils
import fsl.data.image          as fslimage
import fsl.data.gifti          as fslgifti
import fsl.data.vtk            as fslvtk
import fsl.data.freesurfer     as fslfs
import fsl.data.bitmap         as fslbmp
import fsleyes.data.tractogram as tractogram
import fsleyes.displaycontext  as fsldisplay


def guessType(path):
    """Wrapper around the :func:`fsl.data.utils.guessType` function from the
    ``fslpy`` library, augmented to support additional data types supported
    by FSLeyes.

    :arg path: Path to a file to be loaded
    :returns:  Tuple containing:

               - A data type which can be used to load the file, or ``None``
                 if the file is not recognised.
               - A suitable value for the :meth:`.Display.overlayType` for
                 the file, or ``None`` if the file type is not recognised.
               - The file path, possibly modified (e.g. made absolute).
    """

    path  = op.abspath(path)
    dtype = None
    otype = None

    if op.isfile(path):
        if fslpath.hasExt(path.lower(), tractogram.ALLOWED_EXTENSIONS):
            dtype = tractogram.Tractogram

    if dtype is None:
        dtype, path = dutils.guessType(path)

#==================================== numpy support ====================================#
    if dtype is None:
        ext = path.split('.')[-1]
        if ext == 'npy':
            data = np.load(path)
            data = np.array(data, np.float32)
            newpath = PATH+re.sub('[\\s\\.]','',path[0:-3])+'.nii.gz'
            os.makedirs(newpath[0:newpath.rfind('/')], exist_ok=True)
            nib.save(nib.MGHImage(data,np.identity(4),None),newpath)
            dtype, path = dutils.guessType(newpath)
#================================ LayeredArray support =================================#
        elif ext == 'pkl':
            with open(path,'rb') as f:
                data = pickle.load(f).getData()
                data = np.concatenate([np.zeros(data.shape[0:3]+(1,),data.dtype),data],-1)
            data = np.argmax(data, -1)
            data = np.array(data, np.float32)
            newpath = PATH+re.sub('[\\s\\.]','',path[0:-3])+'.nii.gz'
            os.makedirs(newpath[0:newpath.rfind('/')], exist_ok=True)
            nib.save(nib.MGHImage(data,np.identity(4),None),newpath)
            dtype, path = dutils.guessType(newpath)
#=======================================================================================#

    # We need to peek at some images in order
    # to determine a suitable overlay type
    # (e.g. complex images -> "complex")
    if dtype is fslimage.Image:
        img   = fslimage.Image(path)
        otype = fsldisplay.getOverlayTypes(img)[0]
    elif dtype is not None:
        otype = fsldisplay.OVERLAY_TYPES[dtype][0]

    return dtype, otype, path


def overlayName(overlay):
    """Returns a default name for the given overlay. """

    path = overlay.dataSource
    base = op.basename(path)

    if path is not None:
        if isinstance(overlay, fslimage.Nifti):
            return fslimage.removeExt(base)
        elif isinstance(overlay, fslgifti.GiftiMesh):
            return fslpath.removeExt(base, fslgifti.ALLOWED_EXTENSIONS)
        else:
            return base

    if isinstance(overlay, fslimage.Nifti):
        return 'NIfTI image'
    elif isinstance(overlay, fslgifti.GiftiMesh):
        return 'GIfTI surface'
    elif isinstance(overlay, fslfs.FreesurferMesh):
        return 'FreeSurfer surface'
    elif isinstance(overlay, fslvtk.VTKMesh):
        return 'VTK surface'
    elif isinstance(overlay, fslbmp.Bitmap):
        return 'Bitmap'
    elif isinstance(overlay, tractogram.Tractogram):
        return 'Tractogram'
    else:
        return 'Overlay'

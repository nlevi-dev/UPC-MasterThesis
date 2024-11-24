#!/usr/bin/env python

#==================================== numpy support ====================================#
PATH = '/home/levente/fsl/tmp'
import os
import shutil
if os.path.exists(PATH):
    shutil.rmtree(PATH)
    os.mkdir(PATH)
else:
    os.mkdir(PATH)
#=======================================================================================#

if __name__ == '__main__':
    import sys
    import fsleyes.main as main
    sys.exit(main.main())

import sys
from DataHandler import DataHandler

x=int(sys.argv[1])
y=int(sys.argv[2])
z=int(sys.argv[3])
aug_rot = [x,y,z]
print('aug_rot={}'.format(aug_rot))

handler = DataHandler(path='data', space='native', out='logs/augment.log', clear_log=True, cores=-1, aug_rot=aug_rot)
handler.preprocess()
handler.preloadTarget()
handler = DataHandler(path='data', out='logs/augment.log', clear_log=False, cores=-1, aug_rot=aug_rot)
handler.preloadCoords()
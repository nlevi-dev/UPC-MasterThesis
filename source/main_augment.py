import sys
from RecordHandler import RecordHandler

x=int(sys.argv[1])
y=int(sys.argv[2])
z=int(sys.argv[3])
aug_rot = [x,y,z]
print('aug_rot={}'.format(aug_rot))

handler = RecordHandler(path='data', space='native', out='logs/augment.log', clear_log=True, cores=-1, aug_rot=aug_rot)
handler.preprocess()
handler.preloadTarget()
handler = RecordHandler(path='data', out='logs/augment.log', clear_log=False, cores=-1, aug_rot=aug_rot)
handler.preloadCoords()
import numpy as np
import numpy.random as npr
import os

data_dir = '/home/sixd-ailabs/Develop/Human/Hand/diandu/Train'
#anno_file = os.path.join(data_dir, "anno.txt")

size = 12

if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"

with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
    pos = f.readlines()

with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
    part = f.readlines()

with open(os.path.join(data_dir,'%s/landmark_24.txt' %(size)), 'r') as f:
    landmark = f.readlines()

# with open(os.path.join(data_dir, '%s/g_pos_%s.txt' % (size, size)), 'r') as f:
#     g_pos = f.readlines()
#     pos+=g_pos
#
# with open(os.path.join(data_dir, '%s/g_neg_%s.txt' % (size, size)), 'r') as f:
#     g_neg = f.readlines()
#     neg+=g_neg
#
# with open(os.path.join(data_dir, '%s/g_part_%s.txt' % (size, size)), 'r') as f:
#     g_part = f.readlines()
#     part+=g_part
#
# with open(os.path.join(data_dir,'%s/landmark_g_24_aug.txt' %(size)), 'r') as f:
#     g_landmark = f.readlines()
#     landmark+=g_landmark

dir_path = os.path.join(data_dir, 'imglists')
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
if not os.path.exists(os.path.join(dir_path, "%s" %(net))):
    os.makedirs(os.path.join(dir_path, "%s" %(net)))
with open(os.path.join(dir_path, "%s" %(net),"train_%s_landmark.txt" % (net)), "w") as f:
    nums = [len(neg), len(pos), len(part)]
    ratio = [3, 1, 1]
    #base_num = min(nums)
    base_num = 100000
    print(len(neg), len(pos), len(part), base_num)
    if len(neg) > base_num * 3:
        neg_keep = npr.choice(len(neg), size=base_num * 3, replace=True)
        neg_keep=sorted(neg_keep)
    else:
        neg_keep = npr.choice(len(neg), size=len(neg), replace=True)
    pos_keep = npr.choice(len(pos), size=base_num, replace=True)
    part_keep = npr.choice(len(part), size=base_num, replace=True)
    part_keep=sorted(part_keep)
    pos_keep =sorted(pos_keep)
    # neg_keep =range(0,base_num*3)
    # part_keep=range(0,len(part))

    print(len(neg_keep), len(pos_keep), len(part_keep),len(landmark))


    for i in pos_keep:
        line=data_dir+'/'+pos[i]
        f.write(line)
    for i in neg_keep:
        line=data_dir+'/'+neg[i]
        f.write(line)
    for i in part_keep:
        line=data_dir+'/'+part[i]
        f.write(line)
    for item in landmark:
        f.write(item)

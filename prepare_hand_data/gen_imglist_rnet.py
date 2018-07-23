import numpy as np
import numpy.random as npr
import os

data_dir = '/home/sixd-ailabs/Develop/Human/Hand/diandu/Train'
#anno_file = os.path.join(data_dir, "anno.txt")

size = 24

if size == 12:
    net = "PNet"
elif size == 24:
    net = "RNet"
elif size == 48:
    net = "ONet"

with open(os.path.join(data_dir, '%s/pos_%s.txt' % (size, size)), 'r') as f:
    pos = f.readlines()
# with open(os.path.join(data_dir, '%s/pos_%s.txt' % (12, 12)), 'r') as f:
#     pos2 = f.readlines()
# with open(os.path.join(data_dir, '%s/g_pos_%s.txt' % (12, 12)), 'r') as f:
#     pos3 = f.readlines()

with open(os.path.join(data_dir, '%s/neg_%s.txt' % (size, size)), 'r') as f:
    neg = f.readlines()

with open(os.path.join(data_dir, '%s/part_%s.txt' % (size, size)), 'r') as f:
    part = f.readlines()

with open(os.path.join(data_dir, '%s/landmark_%s.txt' % (size, size)), 'r') as f:
    landmark = f.readlines()
  
dir_path = os.path.join(data_dir, 'imglists',"RNet")
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
#write all data
with open(os.path.join(dir_path, "train_%s_landmark.txt" % (net)), "w") as f:
    print len(neg)
    print len(pos)
    # print (len(pos)+len(pos2)+len(pos3))
    print len(part)
    print len(landmark)
    # base_num=len(pos)
    # if len(neg) > base_num * 5:
    #     neg_keep = npr.choice(len(neg), size=base_num * 5, replace=True)
    # else:
    #     neg_keep = npr.choice(len(neg), size=len(neg), replace=True)

    for i in np.arange(len(pos)):
        f.write(pos[i])
    # for i in np.arange(len(pos2)):
    #     f.write(data_dir+'/'+pos2[i])
    # for i in np.arange(len(pos3)):
    #     f.write(data_dir+'/'+pos3[i])
    for i in np.arange(len(neg)):
        f.write(neg[i])
    for i in np.arange(len(part)):
        f.write(part[i])
    for i in np.arange(len(landmark)):
        f.write(landmark[i])

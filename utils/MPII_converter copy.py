import numpy as np
import json
import scipy.io

MPII_KEYPOINT_NUM = 16

SINGLE_ONLY = True

ANNOPOINTS = 'annopoints'

ANNORECT = 'annorect'

POINT = 'point'

ID = 'id'

X = 'x'
Y = 'y'
Z = 'z'

SCALE = 'scale'

OBJPOS = 'objpos'

def is_empty(value_list, key):
    try:
        value_list[key]
    except:
        return True

    if int(len(value_list[key])) > 0:
        return False
    else:
        return True
    
def get_visible(point):
    if is_empty(point, 'is_visible') == True:
        return 0

    visible_array = point['is_visible']

    if visible_array.shape == (1,1):
        is_visible = visible_array[0][0]
    elif visible_array.shape == (1,):
        is_visible = visible_array[0]
    else:
        is_visible = 0

    return is_visible

val_split = 0

def mpii_annotation(mat_path, train_out_path, test_out_path):
    mat_file = scipy.io.loadmat(mat_path)

    mat_data = mat_file['RELEASE']
    
    annolist = mat_data['annolist'][0][0][0]
    img_train = mat_data['img_train'][0][0][0]
    act = mat_data['act'][0][0]
    single_person = mat_data['single_person'][0][0]
    video_list = mat_data['video_list']
    version = mat_data['version']

    train_list = []
    #val_list = []
    test_list = []
    train_val_image_count = 0
    test_image_count = 0
    train_person_count = 0
    val_person_count = 0
    total = len(annolist)

    for i, annotation in enumerate(annolist):
        print(f"{i}/{total}")
        print(annotation.shape)
        img_name = str(annotation['image']['name'][0][0][0])
    
        if is_empty(annotation, ANNORECT):
            continue

        if img_train[i] == 0:
            
            test_image_count += 1
            

        rects = annotation[ANNORECT][0]


        if SINGLE_ONLY:
            single_array = single_person[i][0]
            if 0 in single_array.shape:
                continue
            idx_list = [int(a-1) for a in single_array]
        else:
            idx_list = [int(a) for a in single_array]

        for idx in idx_list:
            rect = rects[idx]
            if (rect is None) or is_empty(rect, ANNOPOINTS):
                continue
                
            points = rect[ANNOPOINTS][POINT][0][0][0]
            points_rect = [[0.,0.,0.,] for j in range(MPII_KEYPOINT_NUM)]
        
            for point in points:
                point_id = point[ID][0][0]
                x = point[X][0][0]
                y = point[Y][0][0]
                is_visible = get_visible(point)
                points_rect[point_id] = list([float(x), float(y), float(is_visible)])

            scale = float(rect[SCALE][0][0])
            objpos = list([float(rect[OBJPOS][X][0][0][0][0]),float(rect[OBJPOS][Y][0][0][0][0])])

            is_validation = float(np.random.rand() < val_split)

            if is_validation == 1.0:
                val_person_count += 1
            else: 
                train_person_count += 1

            annotation_record = {}
            annotation_record['dataset'] = 'MPII'
            annotation_record['img_paths'] = img_name
            annotation_record['is_validation'] = is_validation
            annotation_record['scale_provided'] = scale
            annotation_record['objpos'] = objpos
            annotation_record['joint_self'] = points_rect
            annotation_record['heatbox'] = list([[0.0, 0.0], [0.0, 0.0]])

            if is_empty(rect, 'x1') == False:
                x1 = rect['x1'][0][0]
                y1 = rect['y1'][0][0]
                x2 = rect['x2'][0][0]
                y2 = rect['y2'][0][0]
                annotation_record['heatbox'] = list([[float(x1), float(y1)], [float(x2), float(y2)]])
            #
            if img_train[i] == 1:
                train_val_image_count += 1
                train_list.append(annotation_record)
            else:
                test_image_count += 1
                test_list.append(annotation_record)
            
            #output_list.append(annotation_record)
    if len(train_list) > 0:
        with open(train_out_path, 'w') as file:
            json.dump(train_list, file)
    if len(test_list) > 0:
        with open(test_out_path, 'w') as file:
            json.dump(test_list, file)
    
mat_path = 'D:/MPII-dataset/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat'
train_out_path = 'D:/MPII-dataset/mpii_human_pose_v1_u12_2/MPII_train_annotation.json'
test_out_path = 'D:/MPII-dataset/mpii_human_pose_v1_u12_2/MPII_test_annotation.json'

mpii_annotation(mat_path=mat_path, train_out_path=train_out_path, test_out_path=test_out_path)
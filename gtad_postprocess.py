import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed

from gtad_lib import opts

thumos_class = {
    7 : 'BaseballPitch',
    9 : 'BasketballDunk',
    12: 'Billiards',
    21: 'CleanAndJerk',
    22: 'CliffDiving',
    23: 'CricketBowling',
    24: 'CricketShot',
    26: 'Diving',
    31: 'FrisbeeCatch',
    33: 'GolfSwing',
    36: 'HammerThrow',
    40: 'HighJump',
    45: 'JavelinThrow',
    51: 'LongJump',
    68: 'PoleVault',
    79: 'Shotput',
    85: 'SoccerPenalty',
    92: 'TennisSwing',
    93: 'ThrowDiscus',
    97: 'VolleyballSpiking',
}

def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor

def Soft_NMS(df, nms_threshold=1e-5, num_prop=200):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    # frost: I use a trick here, remove the detection XDD
    # which is longer than 300
    for idx in range(0, len(tscore)):
        if tend[idx] - tstart[idx] >= 300:
            tscore[idx] = 0

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf

def _gen_detection_video(video_name, video_cls, thu_label_id, opt, num_prop=200, topk = 2):

    files = [opt['output']+"/results/" + f for f in os.listdir(opt['output']+"/results/") if
             video_name in f]
    if len(files) == 0:
        # raise FileNotFoundError('Missing result for video {}'.format(video_name))
        print('Missing result for video {}'.format(video_name))
    else:
        # print('Post processing video {}'.format(video_name))
        pass

    dfs = []  # merge pieces of video
    for snippet_file in files:
        snippet_df = pd.read_csv(snippet_file)
        snippet_df['score'] = snippet_df.clr_score.values[:] * snippet_df.reg_socre.values[:]
        snippet_df = Soft_NMS(snippet_df, nms_threshold=opt['nms_thr'])
        dfs.append(snippet_df)
    df = pd.concat(dfs)
    if len(df) > 1:
        df = Soft_NMS(df, nms_threshold=opt['nms_thr'])
    df = df.sort_values(by="score", ascending=False)

    # sort video classification
    video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls))
    unet_classes = [thu_label_id[video_cls_rank[-k-1][1]] + 1 for k in range(topk)]
    unet_scores = [video_cls_rank[-k-1][0] for k in range(topk)]

    fps = result[video_name]['fps']
    num_frames = result[video_name]['num_frames']
    proposal_list = []
    for j in range(min(num_prop, len(df))):
        for k in range(topk):
            tmp_proposal = {}
            tmp_proposal["label"] = thumos_class[int(unet_classes[k])]
            tmp_proposal["score"] = float(round(df.score.values[j] * unet_scores[k], 6))
            tmp_proposal["segment"] = [float(round(max(0, df.xmin.values[j]) / fps, 1)),
                                       float(round(min(num_frames, df.xmax.values[j]) / fps, 1))]
            proposal_list.append(tmp_proposal)
    return {video_name:proposal_list}

def gen_detection_multicore(opt):
    # get video list
    thumos_test_anno = pd.read_csv("./data/thumos_annotations/test_Annotation.csv")
    video_list = thumos_test_anno.video.unique()
    thu_label_id = np.sort(thumos_test_anno.type_idx.unique())[1:] - 1  # get thumos class id
    thu_video_id = np.array([int(i[-4:]) - 1 for i in video_list])  # -1 is to match python index

    # load video level classification
    cls_data = np.load("./data/uNet_test.npy")
    cls_data = cls_data[thu_video_id,:][:, thu_label_id]  # order by video list, output 213x20

    # detection_result
    thumos_gt = pd.read_csv("./data/thumos_annotations/thumos14_test_groundtruth.csv")
    global result
    result = {
        video:
            {
                'fps': thumos_gt.loc[thumos_gt['video-name'] == video]['frame-rate'].values[0],
                'num_frames': thumos_gt.loc[thumos_gt['video-name'] == video]['video-frames'].values[0]
            }
        for video in video_list
    }

    parallel = Parallel(n_jobs=15, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(video_name, video_cls, thu_label_id, opt)
                        for video_name, video_cls in zip(video_list, cls_data ))
    detection_dict = {}
    [detection_dict.update(d) for d in detection]
    output_dict = {"version": "THUMOS14", "results": detection_dict, "external_data": {}}

    with open(opt["output"] + '/detection_result.json', "w") as out:
        json.dump(output_dict, out)

if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    opt["output"] = opt["output"]
    if not os.path.exists(opt["output"]):
        os.makedirs(opt["output"])
    opt_file = open(opt["output"] + "/opts.json", "w")
    json.dump(opt, opt_file)
    opt_file.close()

    print("Detection post processing start")
    gen_detection_multicore(opt)
    print("Detection Post processing finished")

    from evaluation.eval_detection import ANETdetection
    tious = [0.3, 0.4, 0.5, 0.6, 0.7]
    anet_detection = ANETdetection(
        ground_truth_filename='./evaluation/thumos_gt.json',
        prediction_filename=opt["output"] + '/detection_result.json',
        subset='test', tiou_thresholds=tious)
    mAPs, average_mAP = anet_detection.evaluate()
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP))

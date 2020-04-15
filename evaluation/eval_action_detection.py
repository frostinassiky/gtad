import logging
import numpy as np
import pandas as pd

from utils import interpolated_prec_rec
from utils import segment_iou

from joblib import Parallel, delayed

class EvalActionDetection(object):

    def __init__(self, ground_truth_filename=None, 
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 verbose=True):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.ap = None

        self.ground_truth, self.activity_labels = self._import_ground_truth(
            ground_truth_filename)

        if self.verbose:
            logging.info('[INIT] Loaded annotations')
            nr_gt = len(np.unique(self.ground_truth['gt-id']))
            logging.info('\tNumber of ground truth instances: {}'.format(nr_gt))
            logging.info('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth CSV file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_labels : list
            List of the activity classes
        """
        ground_truth = pd.read_csv(ground_truth_filename)
        # remove instances of length <=0 
        ground_truth = ground_truth.loc[ground_truth['t-start'].values < ground_truth['t-end'].values].reset_index(drop=True)

        activity_labels = np.unique(ground_truth['label'])
        ground_truth['gt-id'] = range(len(ground_truth))

        return ground_truth, activity_labels

    def wrapper_compute_average_precision(self, prediction):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_labels)))
        ground_truth_by_label = self.ground_truth.groupby(by='label')

        results = Parallel(n_jobs=len(self.activity_labels))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(l).reset_index(drop=True),
                        prediction=prediction[prediction['label']==l].reset_index(drop=True),
                        tiou_thresholds=self.tiou_thresholds,
                    ) for l in self.activity_labels)
        
        for i in range(len(self.activity_labels)):
            ap[:,i]= results[i]

        return ap

    def evaluate(self, prediction):
        """Evaluates a prediction df. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        logging.info('\tNumber of predictions: {}. Number of unique videos: {}'.format(len(prediction),len(np.unique(prediction['video-name']))))

        self.ap = self.wrapper_compute_average_precision(prediction)
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            logging.info('[RESULTS] Performance on ActivityNet detection task.')
            logging.info('\tmAP: {}'.format(self.mAP))
            logging.info('\tAverage-mAP: {}'.format(self.average_mAP))
        return self.average_mAP

def compute_average_precision_detection(ground_truth, prediction, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-name', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-name, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    gt_id_lst = np.unique(ground_truth['gt-id'].values)
    gt_id_to_index = dict(zip(gt_id_lst, range(len(gt_id_lst))))
    lock_gt = np.ones((len(tiou_thresholds),len(gt_id_to_index))) * -1
    
    npos = float(len(gt_id_lst))

    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))


    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby(by='video-name')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-name'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, gt_id_to_index[this_gt.loc[jdx]['gt-id']]] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, gt_id_to_index[this_gt.loc[jdx]['gt-id']]] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = recall_cumsum * npos / (recall_cumsum * npos + fp_cumsum)
    
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap

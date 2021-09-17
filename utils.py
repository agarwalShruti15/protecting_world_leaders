import os
import pandas as pd
import numpy as np
import pickle
from itertools import combinations
import cv2
from PIL import Image


FEAT_NAME = ['AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r'
            , 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r'
            , 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r'
            , 'AU26_r', 'pose_Rx', 'pose_Rz', 'lip_ver', 'lip_hor']

#get the names of the facial feature combinations listed in global variable FEAT_NAME as 190x2 dimensional np array
def get_all():
    all_feat_comb = list(combinations(range(len(FEAT_NAME)), 2))
    all_feat = np.array([[FEAT_NAME[j[0]], FEAT_NAME[j[1]]] for j in all_feat_comb])
    return all_feat

#return the combined features that can be used to index after correlation
def get_feat_from_file(infile, n):
    return np.array(pd.read_csv(infile)['feat'])[:n]

def save_obj(obj, fldr, name):
    os.makedirs(fldr, exist_ok=True)
    with open(os.path.join(fldr, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#load the pickle file
def load_obj(name):
    if not os.path.exists(name):
        return None
    with open(name, 'rb') as f:
        return pickle.load(f)

#correct landmarks for head rotation
def alignLndmrks_withcsv(csv_file, verbose=False):

    x = np.array(csv_file.loc[:, 'X_0':'X_67'])
    y = np.array(csv_file.loc[:, 'Y_0':'Y_67'])
    z = np.array(csv_file.loc[:, 'Z_0':'Z_67'])

    r_x = np.array(csv_file.loc[:, 'pose_Rx'])
    r_y = np.array(csv_file.loc[:, 'pose_Ry'])
    r_z = np.array(csv_file.loc[:, 'pose_Rz'])

    x_new = x * (np.cos(r_z)*np.cos(r_y))[:, np.newaxis] \
            + y * (np.cos(r_z)*np.sin(r_y)*np.sin(r_x) + np.sin(r_z)*np.cos(r_x))[:, np.newaxis] \
            + z * (np.sin(r_z)*np.sin(r_x) - np.cos(r_z)*np.sin(r_y)*np.cos(r_x))[:, np.newaxis]
    y_new = -x * (np.sin(r_z)*np.cos(r_y))[:, np.newaxis] \
            + y * (np.cos(r_z)*np.cos(r_x) - np.sin(r_z)*np.sin(r_y)*np.sin(r_x))[:, np.newaxis] \
            + z * (np.sin(r_z)*np.sin(r_y)*np.cos(r_x) + np.cos(r_z)*np.sin(r_x))[:, np.newaxis]

    y_new = -y_new

    #for every row find t_x, t_y, theta, and scale
    l_e_x = np.mean(x_new[:, 36:42], axis=1)
    l_e_y = np.mean(y_new[:, 36:42], axis=1)
    r_e_x = np.mean(x_new[:, 42:48], axis=1)
    r_e_y = np.mean(y_new[:, 42:48], axis=1)

    #translate
    x = x_new - l_e_x[:, np.newaxis]
    y = y_new - l_e_y[:, np.newaxis]
    r_e_x = r_e_x - l_e_x
    r_e_y = r_e_y - l_e_y
    l_e_x = l_e_x - l_e_x
    l_e_y = l_e_y - l_e_y

    #rotate theta, assumption r_e_x is positive
    cos_theta = r_e_x / np.sqrt(r_e_x**2 + r_e_y**2)
    sin_theta = np.sqrt(1 - cos_theta**2)
    sin_theta[r_e_y<0] = -sin_theta[r_e_y<0]

    x_new = x * cos_theta[:, np.newaxis] + y * sin_theta[:, np.newaxis]
    y_new = y * cos_theta[:, np.newaxis] - x * sin_theta[:, np.newaxis]
    x = x_new
    y = y_new
    #for every row find t_x, t_y, theta, and scale
    l_e_x = np.mean(x_new[:, 36:42], axis=1)
    l_e_y = np.mean(y_new[:, 36:42], axis=1)
    r_e_x = np.mean(x_new[:, 42:48], axis=1)
    r_e_y = np.mean(y_new[:, 42:48], axis=1)

    #scale
    x = x / r_e_x[:, np.newaxis]
    y = y / r_e_x[:, np.newaxis]
    l_e_x = l_e_x / r_e_x
    l_e_y = l_e_y / r_e_x
    r_e_y = r_e_y / r_e_x
    r_e_x = r_e_x / r_e_x


    if verbose:
        import matplotlib.pyplot as plt
        for i in range(len(l_e_y)):
            plt.clf()
            plt.scatter(x[i, :], y[i, :], c='b', marker='.')
            plt.scatter(l_e_x[i], l_e_y[i], c='r', marker='.')
            plt.scatter(r_e_x[i], r_e_y[i], c='r', marker='.')
            plt.draw()
            plt.pause(0.001) #Note this correction

    out_ar = dict()
    out_ar['x'] = x
    out_ar['y'] = y

    return out_ar


#landmark(,au etc.) dataframe with mouth_h and mouth_v
def combine_lip_opening(in_lndmrk_df):

    cur_feat = alignLndmrks_withcsv(in_lndmrk_df)
    d_hr = ((cur_feat['x'][:, 48]-cur_feat['x'][:, 54])**2)+((cur_feat['y'][:, 48]-cur_feat['y'][:, 54])**2)
    d_vert = ((cur_feat['x'][:, 51]-cur_feat['x'][:, 57])**2)+((cur_feat['y'][:, 51]-cur_feat['y'][:, 57])**2)

    out_df = in_lndmrk_df.copy()
    out_df['lip_hor'] =  d_hr
    out_df['lip_ver'] =  d_vert

    return out_df


"""
    @brief the Pearson Correlation between for all pairs of facial features listed in FEAT_NAME global variable.
        the correlation is computed for overlapping clips with shift shift_win and length vid_len frames
    Note: can return error if the length of the video is smaller than the vid_len variable,
        it rejects the clips where even a single frame is rejected because of less confidence
    @param feat_df: the DataFrame with facial features
        vid_len: length of the clip in frames for computing correlation, default 300 frames
        shift_win: the number of frames to shift to compute next correlation
    @return DataFrame with all the correlation pairs and middle frame
"""
def get_corr_per_frame(feat_df, vid_len=300, shift_win=5):

    cor_feat = get_all()
    col_names = [''.join(x) for x in cor_feat]

    #assert feat_df.shape[0]>(vid_len), 'the video of {} frames should be atleast {}'.format(len(feat_df), vid_len)

    first_col = np.array(cor_feat)[:, 0]; sec_col = np.array(cor_feat)[:, 1]
    frame_rng = np.arange(0, len(feat_df)-vid_len, shift_win)
    tmp_corr = np.zeros((len(frame_rng), len(cor_feat)))
    out_frames = np.zeros((len(frame_rng), ))

    frame_no = np.array(feat_df['frame'])
    i = 0
    for j in frame_rng:
        f, s = np.around(np.array(feat_df[first_col]), 7)[j:j+vid_len, :], np.around(np.array(feat_df[sec_col]), 7)[j:j+vid_len, :]
        # check if they are continous or not
        frms = frame_no[j:j+vid_len]
        if np.sum((frms[1:] - frms[:-1])>1)==0:
            tmp_corr[i, :] = np.corrcoef(f.T, s.T)[0:len(cor_feat), len(cor_feat):].diagonal()
            out_frames[i] = frame_no[int(j+vid_len/2-1)]
            i = i+1

    tmp_corr = tmp_corr[:i, :].copy()
    out_frames = out_frames[:i]
    out_df = pd.DataFrame(tmp_corr, columns=col_names)
    out_df['frame'] = out_frames
    out_df = out_df.dropna()

    return out_df


"""
    @brief extract all the available (426) facial features from OpenFace library. This includes 2D and 3D facial landmarks,
    all Action Units, rotation, translations and confidence etc.
    @param invid_file: the video file to annotate
        open_face_path: path to build/bin folder of OpenFace2.0
        conf_thres: a threshold on the confidence of OpenFace output. 
        If the confidence is less than the threshold we reject the frame. 0.98 is the maximum correlation
        save_csv: whether to save the facial features as csv in the same folder as input
        use_cache: whether to generate the facial features
    @return DataFrame with all the facial features, without less confident frame
"""
def get_facial_features(invid_file, open_face_path, conf_thres=0.98, save_csv=True, use_cache=True):

    open_face_model = os.path.join(open_face_path, 'FeatureExtraction')
    assert os.path.exists(open_face_model), 'OpenFace builds should be kept at path {}'.format(open_face_model)

    root, _ = os.path.splitext(invid_file)
    csv_file = root + '.csv'
    head, _ = os.path.split(invid_file)

    if use_cache and os.path.exists(csv_file):
        out_df = pd.read_csv(csv_file)
        out_df.rename(columns=lambda x: x.strip(), inplace=True)
    
    else:
        cmd = './{} -f {} -q -pose -3Dfp -aus -pdmparams -2Dfp -out_dir {}'.format(open_face_model, invid_file, head)
        os.system(cmd)
        os.remove(root + '_of_details.txt')
        out_df = pd.read_csv(csv_file)
        out_df.rename(columns=lambda x: x.strip(), inplace=True)
        if not save_csv:
            os.remove(csv_file)

    assert out_df is not None, 'the features are not initialized msy be openface not working'

    #clean the facial landmarks
    clean_df = out_df.loc[out_df['confidence']>=conf_thres].copy()
    out_df = combine_lip_opening(clean_df) #combine the transcript and landmark files

    return out_df


"""
    @brief save the openface annotated video in mp4 format
    @param invid_file: the video file to annotate
        out_fldr: the folder to store the annotated video
        open_face_path: path to the build/bin folder of OpenFace2.0
    @return None
"""
def out_tracked_video(invid_file, out_fldr, open_face_path):

    _, tail = os.path.split(invid_file)
    root, _ = os.path.splitext(tail)

    open_face_model = os.path.join(open_face_path, 'FeatureExtraction')
    assert os.path.exists(open_face_model), 'OpenFace builds should be kept at path {}'.format(open_face_model)
    os.makedirs(out_fldr, exist_ok=True)
    cmd = './{} -f {} -q -tracked -oc MP4V -out_dir {}'.format(open_face_model, invid_file, out_fldr)
    print(os.system(cmd))

    #convert to mp4 and delete the avi
    cmd = 'ffmpeg -i {}.avi {}.mp4'.format(os.path.join(out_fldr, root), os.path.join(out_fldr, root))
    os.system(cmd)

    #delete the extra files
    os.remove(os.path.join(out_fldr, root + '_of_details.txt'))
    os.remove(os.path.join(out_fldr, root + '.csv'))
    os.remove(os.path.join(out_fldr, root + '.avi'))

"""
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
"""
def fig2data ( fig ):

    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

"""
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
"""
def fig2img ( fig ):
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return np.array(Image.frombytes( "RGBA", ( w ,h ), buf.tostring()))


"""
    @brief Train the one-class SVM model. 
    The hyper-parameters can be tuned using 5-fold cross-validation if validation dataset is provided  
    @param X_train: the (N, D) numpy array with N instances of D dimensional features
            do_cv: if True then 5-fold cross validation is performed to select gamma and nu parameters of one-class SVM
            default values of gamma and nu are 0.01 and 0.01
            X_val: (N, D) numpy array with N instances of D dimensional features used for cross-validation
            y_val: labels for cross validation
    @return svm_model
"""
def train_ovr(X_train, do_cv=False, X_val=[], y_val=[]):

    from sklearn.model_selection import ParameterGrid
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import OneClassSVM

    # Fit your data on the scaler object
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_scaled = scaler.transform(X_train)

    #init best params to default values
    best_param = {}
    best_param['gamma'] = 0.01
    best_param['kernel'] = 'rbf'
    best_param['nu'] = 0.01

    #SVM model
    clf = OneClassSVM(cache_size=1000)

    # if cross-validation
    if do_cv:

        #if no validation set then optimize for train dataset only
        if len(X_val)==0:
            X_val = X_train.copy()
            y_val = np.ones((len(X_val), ))

        gamma_range = 10. ** np.arange(-5, -1)
        nu = np.linspace(0.01, 0.2, 5)

        #cross validation
        grid = {
            'gamma': gamma_range,
            'kernel': ['rbf'],
            'nu': nu
        }

        best_param = {}; best_acc = 0
        X_test_scaled = scaler.transform(X_val)
        #perform cross-validation on a small train set
        idx = np.random.choice(range(len(X_scaled)), size=4000, replace=False)
        for z in ParameterGrid(grid):

            clf.set_params(**z)
            clf.fit(X_scaled[idx, :])
            y_pred = clf.predict(X_test_scaled)
            y_pred[y_pred<0] = 0

            acc = ((np.sum(np.logical_and(y_pred==0, y_val==y_pred)) /np.sum(y_val==0)) + \
                  (np.sum(np.logical_and(y_pred==1, y_val==y_pred)) /np.sum(y_val==1)))/2
            if acc>best_acc:
                print('gamma:{}, kernel:{}, nu:{}, acc:{} **'.format(z['gamma'],
                                                                     z['kernel'],
                                                                     z['nu'],
                                                                     acc))
                best_param = z
                best_acc = acc
            else:
                print('gamma:{}, kernel:{}, nu:{}, acc:{}'.format(z['gamma'],
                                                                 z['kernel'],
                                                                 z['nu'],
                                                                 acc))

    svm_model = {}
    svm_model['scaler'] = scaler
    svm_model['model'] = clf.set_params(**best_param)
    svm_model['model'].fit(X_scaled)

    return svm_model

"""
    @brief plot the frame and the feature in a pyplot figure and convert it to an RGB image
    @param frame: the video frame
    feat: facial features to plot
    fn: the frame number to draw the red dot
    y_lims: limit of the figure TODO: compute this automatically
    @return np array of the figure as image
"""
def draw_frame_feat(frame, feat, fn, f_rng, y_lims):

    import matplotlib.pyplot as plt

    nr = 1; nc = 2
    w, h = 12, 6
    bgcolor = "#000000"; lbl_color = '#ffffff'

    clipped_feat = feat[f_rng[0]:f_rng[1], :].copy()
    clipped_feat = clipped_feat - np.nanmean(clipped_feat, axis=0, keepdims=True)
    y_lims = [np.nanmin(clipped_feat), np.nanmax(clipped_feat)]

    fig, ax = plt.subplots(nrows=nr, ncols=nc, figsize=(w,h), facecolor=bgcolor)
    ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ax[0].axis('off')

    for j in range(clipped_feat.shape[1]):
        ax[1].plot(clipped_feat[:, j])
        ax[1].plot(fn-f_rng[0], clipped_feat[fn-f_rng[0], j], 'ro', lw=6)

    ax[1].set_xlabel('frames', fontsize=16)
    ax[1].set_ylim(y_lims)
    ax[1].set_xticks(np.arange(0, len(clipped_feat), 50))

    ax[1].set_facecolor(bgcolor)
    for spine in ax[1].spines.values():
        spine.set_edgecolor(lbl_color)
    ax[1].yaxis.label.set_color(lbl_color)
    ax[1].xaxis.label.set_color(lbl_color)
    ax[1].tick_params(axis='x', colors=lbl_color)
    ax[1].tick_params(axis='y', colors=lbl_color)

    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout(pad=0.5, h_pad=None, w_pad=None, rect=None)
    plt.draw()
    im = fig2img(fig)

    plt.close(fig)
    return cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

"""
    @brief save the video with the frames, corresponding facial features. 
    Note: the facial features are not extact but an average over a sequence length avg_len
    @param invid_file: the video file for frames
        facial_feat: all the facial features for corresponding input video 
        comb: the features to visualize with the video frames
        f_rng: the range of the frames to plot, [0, -1] is from start to end
        avg_len: the number of neighboring frames to average for a smooth output
        out_fldr: the folder to store the output video
        y_lims: the limits of the output
    @return None
"""
#TODO: compute the y_lims automatically
def save_corr_video(invid_file, facial_feat, comb, f_rng, avg_len, out_fldr, y_lims=(-1.2, 1.8)):

    selected_feat = facial_feat[comb]
    #standardize
    all_feat = (selected_feat - selected_feat.mean()) / selected_feat.std()

    #open the video reader
    in_video = cv2.VideoCapture(invid_file)
    total_frames = int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
    if f_rng[1] == -1:
        f_rng[1] = total_frames

    #smooth
    fh = int(np.floor(avg_len/2))
    all_feat = all_feat.rolling(avg_len).mean().iloc[avg_len - 1:, :]
    frames = np.array(facial_feat['frame'].iloc[fh - 1:-fh])

    #clip the f_rng
    if frames[0] > f_rng[0]:
        f_rng[0] = frames[0]

    if frames[-1] < f_rng[1]:
        f_rng[1] = frames[-1]

    #get feature for every frame, if not present set to nan
    per_frame_feat = np.zeros((total_frames, 2)) + np.nan
    per_frame_feat[frames, :] = np.array(all_feat)

    #open the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out_vid = None
    out_file = os.path.join(out_fldr, invid_file.split('/')[-1])

    #for each frame write
    ret, frame = in_video.read()  # Extract the frame
    f_n = 0
    while ret:

        if f_n >= f_rng[0] and f_n < f_rng[1]:

            cur_fig_im = draw_frame_feat(frame, per_frame_feat, f_n, f_rng, y_lims)
            if out_vid is None:
                frame_height, frame_width, ch = cur_fig_im.shape
                out_vid = cv2.VideoWriter(out_file,
                                          fourcc,
                                          30, (frame_width,frame_height)) #start a video writer object, #30 is the fps

            out_vid.write(cur_fig_im)

        ret, frame = in_video.read()  # Extract the frame
        f_n += 1

    out_vid.release()

# get the bins to plot using histogram
def get_bins_edges(feats, bins):
    s_counts, s_bin_edges = np.histogram(feats, bins=bins)
    s_counts = s_counts/np.sum(s_counts)
    s_bin_edges = s_bin_edges[:-1]
    return s_counts, s_bin_edges

# Video_file is a 3D array representing pixels X pixels X time
video_file = np.load('reweight_digits.npy')

# Create a copy of the video_fire array
im_focus = video_file.copy()

# Get the number of frames in the video
T = im_focus.shape[2]

# Get the number of rows in the video
N0 = im_focus.shape[0]

# Get the number of columns in the video, leaving out 10 columns
N1 = im_focus.shape[1] - 10

# Create a copy of the extracted frames
low_res = im_focus.copy()

# Get the shape of a single frame
shape_frame = low_res[:, :, 0].shape

# Flatten each frame and store them in a list
video_file_ar = [low_res[:, :, frame].flatten() for frame in range(low_res.shape[2])]

# Create dict_learner object
dict_learner = DictionaryLearning(
    n_components=15, transform_algorithm='lasso_lars', transform_alpha=0.9,
    random_state=402,
)

# List to np.array
video_v = np.vstack(video_file_ar)

# Fit and transform `video_v`
X_transformed = dict_learner.fit(video_v).transform(video_v)
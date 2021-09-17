feature_params = dict(maxCorners=feature_points,qualityLevel=0.3,minDistance=7,
                      blockSize=5,useHarrisDetector=False)
feature_points2 = cv2.goodFeaturesToTrack(image=color_image2,mask=None,**feature_params)
feature_points3 = cv2.goodFeaturesToTrack(image=color_image3,mask=None,**feature_params)
    
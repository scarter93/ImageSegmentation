root = pwd;

ipath = ['-I', fullfile(root, 'include')];
lpath = fullfile(root, 'libs');


lib1 = fullfile(lpath, 'opencv_calib3d2411.lib');
lib2 = fullfile(lpath, 'opencv_contrib2411.lib');
lib3 = fullfile(lpath, 'opencv_core2411.lib');
lib4 = fullfile(lpath, 'opencv_features2d2411.lib');
lib5 = fullfile(lpath, 'opencv_flann2411.lib');
lib6 = fullfile(lpath, 'opencv_gpu2411.lib');
lib7 = fullfile(lpath, 'opencv_highgui2411.lib');
lib8 = fullfile(lpath, 'opencv_imgproc2411.lib');
lib9 = fullfile(lpath, 'opencv_legacy2411.lib');
lib10 = fullfile(lpath, 'opencv_ml2411.lib');
lib11 = fullfile(lpath, 'opencv_nonfree2411.lib');
lib12 = fullfile(lpath, 'opencv_objdetect2411.lib');
lib13 = fullfile(lpath, 'opencv_ocl2411.lib');
lib14 = fullfile(lpath, 'opencv_photo2411.lib');
lib15 = fullfile(lpath, 'opencv_stitching2411.lib');
lib16 = fullfile(lpath, 'opencv_superres2411.lib');
lib17 = fullfile(lpath, 'opencv_ts2411.lib');
lib18 = fullfile(lpath, 'opencv_video2411.lib');
lib19 = fullfile(lpath, 'opencv_videostab2411.lib');

mex('kmeansMex.cpp', ipath, lib1, lib2, lib3, lib4, lib5, lib6, lib7, lib8, lib9, lib10, lib11, lib12, lib13, lib14, lib15, lib16, lib17, lib18, lib19)
roslaunch realsense2_camera rs_camera.launch & sleep 10;
roslaunch mavros px4.launch & sleep 10;
roslaunch vins run_avata_vins.launch
wait;

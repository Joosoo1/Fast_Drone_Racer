#include <iostream>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <unistd.h>
#include <ros/ros.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <Eigen/Dense>
#include <airsim_ros/Circle.h>
#include <airsim_ros/CirclePoses.h>
#include <bbox_ex_msgs/BoundingBox.h>
#include <bbox_ex_msgs/BoundingBoxes.h>

ros::Publisher pose_pub;//gates
ros::Publisher target_pub;//targets;
ros::Subscriber pose_sub;
ros::Subscriber target_pose;

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "global_circle_sub_node");
    ros::NodeHandle nh("~");
     
    pose_pub = nh.advertise<airsim_ros::CirclePoses>("gate_poses)", 10);

    int client_fd,ret;
    struct sockaddr_in _addr;
    
    int broadcastEnable = 1;
    memset(&_addr, 0, sizeof(_addr));
    _addr.sin_family = AF_INET;
    _addr.sin_port = htons(5555);
    client_fd = socket(AF_INET, SOCK_DGRAM, 0);
    setsockopt(client_fd, SOL_SOCKET, SO_BROADCAST, (char*)&broadcastEnable, sizeof(broadcastEnable));
    bind(client_fd, (struct sockaddr*)&_addr, sizeof(_addr));
    socklen_t len;
    int count;
    struct sockaddr_in clent_addr;
    std::cout<<"2";

    ros::Rate r(1);
    while(ros::ok())
    {
        airsim_ros::CirclePoses circles_pub;
        geometry_msgs::PoseArray gate_poses;
        gate_poses.header.stamp = ros::Time::now();
        gate_poses.header.frame_id = "world";
        double data[70];
        std::cout<<"1";
        memset (data, 0, 70*sizeof(double));
        int count= recvfrom(client_fd, data, 70 * sizeof(double), 0, (struct sockaddr*)&clent_addr, &len);
        if(count == -1)
        {
            std::cout <<"recieve data fail!"<<std::endl;
            return 0;
        }
        std::cout<<"\nGet data:"<<std::endl;
        for(int i = 0; i<70; i= i+7)
        {   
            geometry_msgs::Pose gate_pose;
            gate_pose.position.x = data[i];
            gate_pose.position.y = data[i+1];
            gate_pose.position.z = data[i+2];
            gate_pose.orientation.w = data[i+3];
            gate_pose.orientation.x = data[i+4];
            gate_pose.orientation.y = data[i+5];
            gate_pose.orientation.z = data[i+6];
            Eigen::Quaterniond quat(gate_pose.orientation.w,gate_pose.orientation.x,gate_pose.orientation.y,gate_pose.orientation.z);
            Eigen::Matrix3d rotation_mat = quat.toRotationMatrix();
            Eigen::Vector3d eular = rotation_mat.eulerAngles(0, 1, 2);
            double yaw = eular(2);
            std::cout<<"yaw:"<<yaw<<std::endl;
            airsim_ros::Circle circle_pub;
            int j = 0;
            circle_pub.index = j;
            circle_pub.position.x = gate_pose.position.x;
            circle_pub.position.y = gate_pose.position.y;
            circle_pub.position.z = gate_pose.position.z;
            circle_pub.yaw = yaw;
            
            circles_pub.header.stamp = ros::Time::now();
            circles_pub.header.frame_id = "world";
            circles_pub.poses.push_back(circle_pub);
            j++;
        
            //gate_poses.poses.push_back(gate_pose);
                      
        }

        pose_pub.publish(circles_pub);
       
        r.sleep();
        ros::spinOnce();

    }
    return 0;
}
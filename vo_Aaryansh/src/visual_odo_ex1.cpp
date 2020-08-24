#include"vo_features.h"
#include<fstream>

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000


//cameraMatrix
Mat K = (Mat_<double>(3,3) << 543.3327734182214, 0.00, 489.02536042247897, 0.00, 542.398772982566, 305.38727712002805, 0.00, 0.00, 1.00);

//distCoeff Mat
Mat distCoeff = (Mat_<double>(1,4) << -0.1255945656257394, 0.053221287232781606, 9.94070021080493e-05,9.550660927242349e-05);

Mat orig_pose = (Mat_<double> (4,1) <<0,0,0,1);
Mat null = (Mat_<double> (1,3) << 0,0,0);
Mat one = (Mat_<double> (1,1) << 1);

//initial frame
Mat R_i = (Mat_<double>(3,3) << 1,0,0,0,1,0,0,0,1);
Mat t_i = (Mat_<double>(3,1) << 0,0,0);
Mat R_f = R_i;
Mat t_f = t_i;


//function to compute distance
double rel_dis(vector<float> c1, vector<float> c2){

	return sqrt((c2[0]-c1[0])*(c2[0]-c2[0]) + (c2[1]-c1[1])*(c2[1]-c1[1]) + (c2[2]-c1[2])*(c2[2]-c1[2])) ;
	
}

//function to compute the relative scale by triangulation
//Input->pairs of subsequent images and their feature vector

double RelativeScale(Mat img1, Mat img2, Mat img3,vector<Point2f> points1,
					vector<Point2f> points2,vector<Point2f> points3,vector<Point2f> points4,
					Mat R_1,Mat t_1,Mat R_2,Mat t_2){

	Mat R_a,t_a;
	Mat R_b,t_b;

	Mat cam0,cam1;
	Mat cam2;
	int i=0;
	double norm;

	hconcat(R_f,t_f,cam0);  //cam0 -> camera projection matrix

	R_a=R_1*R_f;
	t_a=t_1+t_f;
	norm=sqrt(t_a.at<double>(0)*t_a.at<double>(0) + t_a.at<double>(1)*t_a.at<double>(1) + t_a.at<double>(2)*t_a.at<double>(2));
	t_a=(1/norm)*t_a;

	hconcat(R_a,t_a,cam1); //cam1 ->camera projection matrix


	R_b=R_2*R_a;
	t_b=t_2 + t_a;
	norm=sqrt(t_b.at<double>(0)*t_b.at<double>(0) + t_b.at<double>(1)*t_b.at<double>(1) + t_b.at<double>(2)*t_b.at<double>(2));
	t_b=(1/norm)*t_b;


	hconcat(R_b,t_b,cam2);
	//float det_R=determinant(R_1);
	//cout<<"R_1: "<<R_1<<endl;
	//cout<<det_R<<endl;

	vector<float> v1={0,0,0};
	vector<float> v2={0,0,0};
	vector<float> v3={0,0,0};
	vector<float> v4={0,0,0};
	
	if((points1.size()!=points2.size())||(points3.size()!=points4.size())){
		cout<<"size not same"<<endl;
	}

	Mat pnts3D_1(3,points1.size(),CV_64FC1);
	triangulatePoints(cam0,cam1,points1,points2,pnts3D_1);    //triangulation of first pair

	Mat pnts3D_2(3,points2.size(),CV_64F);
	triangulatePoints(cam1,cam2,points3,points4,pnts3D_2);   //trinagulation for second one

	double scale,numerator,denominator;
	scale=0.0;

	for(i=0;((i<10) && (i+1< points1.size()) && (i+1< points2.size())); i++){

		v1[0]=pnts3D_1.at<float>(0,i);
		v1[1]=pnts3D_1.at<float>(1,i);
		v1[2]=pnts3D_1.at<float>(2,i);

		v2[0]=pnts3D_1.at<float>(0,i+1);
		v2[1]=pnts3D_1.at<float>(1,i+1);
		v2[2]=pnts3D_1.at<float>(2,i+1);

		v3[0]=pnts3D_2.at<float>(0,i);
		v3[1]=pnts3D_2.at<float>(1,i);
		v3[2]=pnts3D_2.at<float>(2,i);

		v4[0]=pnts3D_2.at<float>(0,i+1);
		v4[1]=pnts3D_2.at<float>(1,i+1);
		v4[2]=pnts3D_2.at<float>(2,i+1);

		numerator=rel_dis(v1,v2);
		denominator=rel_dis(v3,v4);

		scale+=(numerator/denominator);
	}

	R_f=R_b;
	t_f=t_b;

	//taking mean for scale
	if(i>0){
		return scale/i;
	}

	else{
		cout<<"improper scale"<<endl;
	}

}


int main(int argc,char** argv){
	 Mat prev_pose = orig_pose;
	 Mat cur_pose;

	 Mat img_1, img_2, img1, img2;
	 

	 ofstream ofile;
	 ofile.open("data/pos.txt");

	 double scale=1.00;

	 char text[100];
	 int fontFace = FONT_HERSHEY_PLAIN;
	 double fontScale=1;
	 int thickness =1;
	 cv::Point textOrg(10,50);

	 Mat img_1_c=imread("../../aqualoc_dataset/images_sequence/frame000000.png");  //step1:Takes two frames
	 Mat img_2_c=imread("../../aqualoc_dataset/images_sequence/frame000001.png");

	 


	 if(!img_1_c.data || !img_2_c.data){
	 	cout<<"Error reading images!"<<endl; return -1;}

	 undistort(img_1_c,img1,K,distCoeff);         //step2: Undistortion
	 undistort(img_2_c,img2,K,distCoeff);

	 cvtColor(img1,img_1,COLOR_BGR2GRAY);
	 cvtColor(img2,img_2,COLOR_BGR2GRAY);


	 vector<Point2f> points1_1,points2_1;
	 vector<Point2f> point1_2,points2_2;
	 featureDetection(img_1,points1_1);          //step3: Featuredetection

	 vector<uchar> status_1;
	 featureTracking(img_1,img_2,points1_1,points2_1,status_1);  //Step4: FeatureTracking

	 Mat E_1, R_1, t_1, mask_1;	
	 Mat E_2,R_2,t_2,mask_2;
	 Mat t_p,R_p;
	 Mat t_f,R_f;
	 Mat T_h,T_v;
	 Mat T;
	 Mat T_main = (Mat_<double>(4,4) << 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);


	 E_1=findEssentialMat(points2_1, points1_1, K, RANSAC, 0.999, 1.0, mask_1);
	 recoverPose(E_1, points2_1, points1_1, K, R_1, t_1, mask_1);    //step5: Essential Matrix


	 Mat prevImage =img_2;
	 Mat prevLastImage = img_1;
	 Mat currImage;
	 vector<Point2f> prevFeatures =points2_1;
	 vector<Point2f> currFeatures;

	 char filename[100];
	 

	 clock_t begin=clock();

	 namedWindow("Underwater Camera", WINDOW_AUTOSIZE);
	 namedWindow("Trajectory", WINDOW_AUTOSIZE);

	 Mat traj = Mat::zeros(600, 600, CV_8UC3);

	 //Loop to do above tasks on further frames
	 for(int numFrame=2; numFrame<MAX_FRAME; numFrame++){
	 	sprintf(filename,"../../aqualoc_dataset/images_sequence/frame%06d.png", numFrame);

	 	Mat currImage_c=imread(filename);
	 	Mat currImage_t;
	 	undistort(currImage_c,currImage_t,K,distCoeff);
	 	cvtColor(currImage_t,currImage, COLOR_BGR2GRAY);
	 	vector<uchar> status;
	 	featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);   //Step4: Feature Tracking 


	 	E_2=findEssentialMat(currFeatures, prevFeatures, K, RANSAC, 0.999, 1.0, mask_2); //Step5: Essential Matrix
	 	recoverPose(E_2, currFeatures, prevFeatures, K, R_2, t_2, mask_2);

	 	scale=RelativeScale(prevLastImage,prevImage,currImage,points1_1,points2_1,
	 						prevFeatures,currFeatures,R_1,t_1,R_2,t_2);   //Step6: Rescaling

	 	Mat prevPts(2,prevFeatures.size(), CV_64F), currPts(2,currFeatures.size(), CV_64F);


   for(int i=0;i<prevFeatures.size();i++)	{   
  		prevPts.at<double>(0,i) = prevFeatures.at(i).x;
  		prevPts.at<double>(1,i) = prevFeatures.at(i).y;

  		currPts.at<double>(0,i) = currFeatures.at(i).x;
  		currPts.at<double>(1,i) = currFeatures.at(i).y;
    }


		/*t_p=t_i + (R_i*t_1);
		R_p=(R_1*R_i);

		t_f=t_p + scale*(R_p*t_2);
		R_f=(R_2*R_p);*/

    	//scale=0.2;
		t_1=scale*(t_1);   //Scaling unit translation vector

		hconcat(R_1,t_1,T_h);
		hconcat(null,one,T_v);
		vconcat(T_h,T_v,T);     //Rigid Body Transformation Matrix(T)

		T_main=T*T_main;        

		cur_pose=(T_main*prev_pose);
		cout<<"T_main"<<T_main<<endl;
    	   


    if (prevFeatures.size() < MIN_NUM_FEAT)	{
      //cout << "Number of tracked features reduced to " << prevFeatures.size() << endl;
      //cout << "trigerring redection" << endl;
 		featureDetection(prevImage, prevFeatures);
      featureTracking(prevImage,currImage,prevFeatures,currFeatures, status);

	 }

	prevImage = currImage.clone();
	points1_1=prevFeatures;
	points2_1=currFeatures;
    prevFeatures = currFeatures;

    int x = int(cur_pose.at<double>(0)) + 300;
    int y = int(cur_pose.at<double>(2)) + 100;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);

    rectangle( traj, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), FILLED);
    sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", cur_pose.at<double>(0), cur_pose.at<double>(1), cur_pose.at<double>(2));
    putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness, 8);

    if(numFrame%20==0 || numFrame==2)
    {ofile<<numFrame<<" "<<cur_pose.at<double>(0)<<"       "<<cur_pose.at<double>(1)<<"       "<<cur_pose.at<double>(2)<<endl;}

    cout<<"FrameId"<<numFrame<<endl;

	//R_i=R_1.clone();
	//t_i=t_1.clone();

	R_1=R_2.clone();
	t_1=t_2.clone();
	prev_pose=cur_pose;

    imshow( "Underwater Camera", currImage_c );
    imshow( "Trajectory", traj );

    waitKey(1);

}

	clock_t end = clock();
  	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  	cout << "Total time taken: " << elapsed_secs << "s" << endl;

  	//cout << R_f << endl;
  	//cout << t_f << endl;


  	ofile.close();

  	return 0;	

}
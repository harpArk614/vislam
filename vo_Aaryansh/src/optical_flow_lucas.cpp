#include<iostream>
#include<sstream>
#include<iomanip>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/videoio.hpp>
#include<opencv2/video.hpp>

using namespace std;
using namespace cv;


int main(int argc,char** argv){

	vector<Scalar> colors;
	RNG rng;
	for(int i=0;i<100;i++)
	{
		int r=rng.uniform(0,256);
		int g=rng.uniform(0,256);
		int b=rng.uniform(0,256);
		colors.push_back(Scalar(r,g,b));
	}

	Mat old_frame, old_gray;
	vector<Point2f> p0,p1;

	old_frame=imread("../../aqualoc_dataset/images_sequence/frame000000.png");
	cvtColor(old_frame,old_gray,COLOR_BGR2GRAY);

	goodFeaturesToTrack(old_gray,p0,20,0.3,7,Mat(),7,false,0.04);

	Mat mask=Mat::zeros(old_frame.size(),old_frame.type());
	int i=1;

	while(true){

		Mat frame, frame_gray;
		//capture>>frame;
		stringstream strbf;
		strbf<<"../../aqualoc_dataset/images_sequence/frame"<<setw(6)<<setfill('0')<<i<<".png";
		frame=imread(strbf.str());

		if (frame.empty())
			break;
		cvtColor(frame,frame_gray,COLOR_BGR2GRAY);

		vector<uchar> status;
		vector<float> err;

		TermCriteria criteria=TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS),10,0.03);
		calcOpticalFlowPyrLK(old_gray, frame_gray, p0,p1,status,err,Size(15,15),2,criteria);

		vector<Point2f> good_new;
		for(uint i=0; i<p0.size();i++)
		{
			if(status[i]==1){
				good_new.push_back(p1[i]);
				line(mask,p1[i],p0[i],colors[i],2);
				circle(frame,p1[i],5,colors[i],-1);

			}
		}

		Mat img;
		add(frame,mask,img);

		imshow("Frame",img);

		cout<<"FrameId"<<i<<endl;

		int keyboard=waitKey(30);
		if(keyboard=='q' || keyboard==27)
			break;

		old_gray=frame_gray.clone();
		p0=good_new;
		i++;
	}
}

#include <opencv2\core\core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <stack>
#include <stdio.h>
#include <tuple>

#include "PlaceCircle.hpp"

using namespace std;
using namespace cv;
int main() {
	//tuple < string, int, int
	vector<tuple<string, int, int, string> > vecIn;

	vecIn.push_back(make_tuple(".\\img_2.bmp", 15, 2,".\\out_2.bmp"));
	vecIn.push_back(make_tuple(".\\img_4.bmp", 15, 4,".\\out_4.bmp"));
	vecIn.push_back(make_tuple(".\\img_9.bmp", 15, 9,".\\out_9.bmp"));
	vecIn.push_back(make_tuple(".\\img_10.bmp", 15, 10,".\\out_10.bmp"));

	for (auto f : vecIn) {
		CPlaceCircle bp;
		Mat img = imread(get<0>(f)),ans;
		bp.test(img, get<1>(f), 1, 0.93, ans, get<2>(f));
		imshow("img", img);
		imshow("ans",ans);
		imwrite(get<3>(f).c_str(), ans);
		waitKey();
	}
	return 0;
}
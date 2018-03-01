#ifndef __BRICK_PARTITION__
#define __BRICK_PARTITION__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <math.h>
#include <stack>
#include <set>

using namespace std;
using namespace cv;


class CPlaceCircle {
private:
	int Q = 6; // quantitie
	int T = 3; // time
	int time_limit;
	int est_ans;
	int m_answer_color = 87;
	int max_group_size = 200;
	int sml_group_size[20], st_of_sml_group[20];
	int m_max_size_x = 1000;
	int m_max_size_y = 1000;
	double tolerance, overlap_dist;
	int m_max_size_area = m_max_size_x * m_max_size_y;
	int single_cir_size;
	int num_of_cir, debug = 1;
	int found_key;
	Mat Gray_Img;
	vector < pair<int, int> > vec_of_legp_all, vec_of_legp, vec_of_ans, vec_of_last_ans, vec_of_cir;
	set< int >valid_pts[20];
	vector<int> vec_of_temp_ans, vec_of_overlap[250];
	stack<int> stk_of_kill[20];
	time_t start_time, now_time;
	int answer_amount, belong[250];
	int prev_node[255],next_node[255],alive[255];
	double m_EPS = 1e-3;
	inline double dist_of_2pts(pair<int, int> pa, pair<int, int> pb) {
		return (pa.first - pb.first) * (pa.first - pb.first) + (pa.second - pb.second) * (pa.second - pb.second);
	}
	inline void fill_cir(int x, int y) {
		for (auto f : vec_of_cir) {
			if (x + f.first >= 0 && x + f.first < Gray_Img.rows && y + f.second >= 0 && y + f.second < Gray_Img.cols) {
				Gray_Img.at<uchar>(x + f.first, y + f.second) = m_answer_color;
			}
		}
	}
	inline void make_vec_of_cir() {
		for (int i = -single_cir_size; i <= single_cir_size; i++) {
			for (int j = -single_cir_size; j <= single_cir_size; j++) {
				if (dist_of_2pts(make_pair(0, 0), make_pair(i, j)) <= single_cir_size*single_cir_size) {
					vec_of_cir.push_back(make_pair(i, j));
				}
			}
		}
	}
	inline void init() {
		vec_of_legp_all.clear();
		vec_of_legp.clear();
		vec_of_temp_ans.clear();
		vec_of_ans.clear();
		vec_of_cir.clear();
		answer_amount = 0;
		num_of_cir = 0;
	}
	void found_solution(int r) {
		if (found_key) return;
		if (r == est_ans) {
			cout << "found it" << endl;
			for (int i = 0; i<r; i++) {
				vec_of_ans.push_back(vec_of_legp[vec_of_temp_ans[i]]);
			}
			found_key = 1;
			return;
		}
		time(&now_time);
		if (difftime(now_time, start_time) >= time_limit) return;
		int flag = 250 - r - 1;
		for (int f = next_node[flag]; f != flag; f = next_node[f]) {
			
			for (auto f2 : vec_of_overlap[f]) {
				if (alive[f2]) {

					stk_of_kill[r].push(f2);
					alive[f2] = 0;
					prev_node[next_node[f2]] = prev_node[f2];
					next_node[prev_node[f2]] = next_node[f2];
					//valid_pts[belong[f2]].erase(f2);
				}
			}
			vec_of_temp_ans.push_back(f);
			found_solution(r + 1);
			vec_of_temp_ans.pop_back();

			while (!stk_of_kill[r].empty()) {
				alive[stk_of_kill[r].top()] = 1;
				next_node[prev_node[stk_of_kill[r].top()]] = prev_node[next_node[stk_of_kill[r].top()]] = stk_of_kill[r].top();
				//valid_pts[belong[stk_of_kill[r].top()]].insert(stk_of_kill[r].top());
				stk_of_kill[r].pop();
			}
			if (found_key) return;
		}
		
	}
public:
	inline CPlaceCircle() {
		return;

	}
	inline ~CPlaceCircle() {
		return;
	}
	inline int test(cv::Mat &brickSegMat, double radius, int step, double Tolerance, Mat &ans, int target_circle_num)
	{
		int group_size;
		found_key = 0;
		single_cir_size = radius;
		est_ans = target_circle_num;
		tolerance = Tolerance;
		
		time_limit = T;
		make_vec_of_cir();
		if (brickSegMat.channels() == 3)
		{
			cvtColor(brickSegMat, Gray_Img, CV_BGR2GRAY);
		}
		else
		{
			Gray_Img = brickSegMat.clone();
		}
		Mat element = getStructuringElement(MORPH_ELLIPSE, cv::Size(single_cir_size * 2 + 1, single_cir_size * 2 + 1));
		Mat candidateMat, pixMat;


		cv::threshold(Gray_Img, pixMat, 100, 1, THRESH_BINARY);

		pixMat.convertTo(pixMat, CV_32S);

		filter2D(pixMat, candidateMat, -1, element, Point(-1, -1), 0, BORDER_DEFAULT);

		//cv::threshold(candidateMat, candidateMat, (double)vec_of_cir.size()*m_Tolerance / 100, 1, THRESH_BINARY);
		cv::inRange(candidateMat, (double)vec_of_cir.size()*Tolerance, (double)vec_of_cir.size() * 10, candidateMat);

		//num_of_cir = 0;
		vec_of_legp_all.clear();
		for (int i = 0; i < Gray_Img.rows; i++) {
			for (int j = 0; j < Gray_Img.cols; j++) {
				if (candidateMat.at<uchar>(i, j)) {
					vec_of_legp_all.push_back(make_pair(i, j));
				}
			}
		}


		if (debug) {
			cout << "legp size = " << vec_of_legp_all.size() << endl;
		}

		int bs_p, bs_q, bs_mid;
		bs_p = Tolerance * single_cir_size * single_cir_size * 4;
		bs_q = 1.5 * single_cir_size * single_cir_size * 4;
		while ( bs_p <= bs_q || (int)(vec_of_last_ans.size()) == 0) {
			bs_mid = (bs_p + bs_q) / 2;
			found_key = 0;
			vec_of_ans.clear();
			// temporary set n as 80 
			group_size = min(max_group_size, (int)vec_of_legp_all.size());
			vec_of_legp.resize(group_size);
			st_of_sml_group[0] = 0;
			for (int i = 0; i < est_ans; i++) {
				sml_group_size[i] = group_size / est_ans;
				if (i < group_size - group_size / est_ans * est_ans) sml_group_size[i] ++;
				st_of_sml_group[i + 1] = st_of_sml_group[i] + sml_group_size[i];
			}

			/*for (int i = 0; i < est_ans; i++) {
				valid_pts[i].clear();
				for (int j = 0; j < sml_group_size[i]; j++) {
					belong[st_of_sml_group[i] + j] = i;
					valid_pts[i].insert(st_of_sml_group[i] + j);
				}
			}*/
			for (int i = 0; i < est_ans; i++) {
				for (int j = 0; j < sml_group_size[i]; j++) {
					belong[st_of_sml_group[i] + j] = i;
					if (j == 0) {
						next_node[250 - i - 1] = st_of_sml_group[i] + j;
						prev_node[st_of_sml_group[i] + j] = 250 - i - 1;
					}
					else {
						next_node[st_of_sml_group[i] + j - 1] = st_of_sml_group[i] + j;
						prev_node[st_of_sml_group[i] + j] = st_of_sml_group[i] + j - 1;
					}
					if (j == sml_group_size[i] - 1) {
						prev_node[250 - i - 1] = st_of_sml_group[i] + j;
						next_node[st_of_sml_group[i] + j] = 250 - i - 1;
					}
					else {
						prev_node[st_of_sml_group[i] + j + 1] = st_of_sml_group[i] + j;
						next_node[st_of_sml_group[i] + j] = st_of_sml_group[i] + j + 1;
					}
				}
			}
			time(&start_time);
			while (1) {
				srand(time(0));
				random_shuffle(vec_of_legp_all.begin(), vec_of_legp_all.end());
				for (int i = 0; i < group_size; i++) {
					vec_of_legp[i] = vec_of_legp_all[i];
					alive[i] = 1;
				}

				for (int i = 0; i<group_size; i++) {
					vec_of_overlap[i].clear();
					for (int j = st_of_sml_group[belong[i] + 1]; j<group_size; j++) {
						if (dist_of_2pts(vec_of_legp[i], vec_of_legp[j]) <= bs_mid) {
							vec_of_overlap[i].push_back(j);
						}
					}
				}
				found_solution(0);
				
				if (found_key) {
					break;
				}
				time(&now_time);
				if (difftime(now_time, start_time) >= time_limit) break;
			}
			if (found_key) {
				cout << "found ans : " << endl;
				for (auto f : vec_of_ans) {
					cout << "(" << f.first << ',' << f.second << ")" << endl;
				}
				if ((int)(vec_of_last_ans.size()) == 0) {
					vec_of_last_ans.resize(vec_of_ans.size());
					for (int t = 0; t < vec_of_ans.size(); t++) {
						vec_of_last_ans[t] = vec_of_ans[t];
					}
				}
				else {
					int min_dist_now, min_dist_last;
					min_dist_now = 0x7FFFFFFF;
					min_dist_last = 0X7FFFFFFF;
					for (int di = 0; di < vec_of_ans.size(); di++) {
						for (int dj = di + 1; dj < vec_of_ans.size(); dj++) {
							min_dist_now = min(int(dist_of_2pts(vec_of_ans[di], vec_of_ans[dj])), min_dist_now);
							min_dist_last = min(int(dist_of_2pts(vec_of_last_ans[di], vec_of_last_ans[dj])), min_dist_last);
						}
					}
					if (min_dist_now > min_dist_last) {
						for (int t = 0; t < vec_of_ans.size(); t++) {
							vec_of_last_ans[t] = vec_of_ans[t];
						}
					}
				}
				bs_p = bs_mid + 1;
			}
			else {
				cout << "didn't found answer" << endl;
				bs_q = bs_mid - 1;
			}
		}

		//fill_cir();
		//cout << "best answer = " << answer;
		num_of_cir = 0;
		for (auto f : vec_of_last_ans) {
			fill_cir(f.first, f.second);
			//ptList.push_back(Point2f(f.second, f.first));
			num_of_cir++;
		}
		ans = Gray_Img.clone();
		return 0;
	}

};

#endif
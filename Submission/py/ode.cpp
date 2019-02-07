/*

Ray tracer written in c++ to speed up tracing. Bad idea too many problems to
fix.

*/


#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

vector<double> linspace(double a, double b, double c) {
    vector<double> array;
    while(a <= c) {
        array.push_back(a);
        a += b;         // could recode to better handle rounding errors
    }
    return array;
}

class Plane {
  public:
    Plane();
    double r[3] = {1.0,0.0,0.0};
    double normal[3] = {1.0,0.0,0.0};
    double w = 1;
    double h = 1*9/16;
    double p1[2] = {r[1]-w/2, r[2]-h/2};
    double p2[2] = {r[1]+w/2, r[2]+h/2};
    vector<double> p_x = linspace(p1[0],p2[0],640);
    vector<double> p_y = linspace(p1[1],p2[1],448);

    double find_nearest(vector<double> arr, double val){
      double ind = 0;
      double min = 500;
      for (int i = 0; i < arr.size(); i++){
        if (abs(arr[i]-val) < min) {
          min = abs(arr[i]-val);
        }
      }
      return min;
    }

    double *get_ind(double x, double y){
      double x_ind = find_nearest(p_x,x);
      double y_ind = find_nearest(p_y,y);
      double arr[2] = {x_ind,y_ind};
      return arr;
    }
};

double *RK4f(double y[], double h2[]){
  double f[6] = {y[3],y[4],y[5],0,0,0};
  f[3] = -1.5 * h2[0] * y[0] / pow(y[0],5);
  f[4] = -1.5 * h2[1] * y[1] / pow(y[1],5);
  f[5] = -1.5 * h2[2] * y[2] / pow(y[2],5);
  return f;
}

vector<double> add_arr(double arr_1[], double arr_2[]){
  vector<double> arr[arr_1.size()];
  for (int i = 0; i < arr_1.size(), i++){
    arr[i] = arr_1[i] + arr_2[i];
  }
  return arr;
}

double *mult_arr(double arr_1[], double arr_2[]){
  double *arr[arr_1.size()];
  for (int i = 0; i < arr_1.size(), i++){
    arr[i] = arr_1[i]*arr_2[i];
  }
  return arr;
}

double *rk4(double y[], double h2[], double h){
  double *k1 = RK4f(y,h2);
  double step[3] = {0.5*h, 0.5*h, 0.5*h};
  double h_arr[3] = {h,h,h};
  double two[3] = {2,2,2};
  double h6[3] = {h/6,h/6,h/6};
  double *k2 = RK4f(add_arr(y,mult_arr(step,k1)),h2);
  double *k3 = RK4f(add_arr(y,mult_arr(step,k2)),h2);
  double *k4 = RK4f(add_arr(y,mult_arr(h_arr,k3)),h2);
  double *val = mult_arr(h6,add_arr(add_arr(k1,mult_arr(two,k2)),add_arr(mult_arr(two,k3),k4)));
  return val;
}
vector<double> cross(vector<double> vect_A, vector<double> vect_B){
  vector<double> cross_P[3];
  cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
  cross_P[1] = vect_A[0] * vect_B[2] - vect_A[2] * vect_B[0];
  cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];
  return cross_P;
}
double norm(double *arr){
  double tot = 0;
  for (int i = 0; i < 3; i++){
    tot += pow(arr[i],2);
  }
  tot = pow(tot,0.5);
  return tot;
}

double sqrsum(vector<double> arr){
  double sum = 0;
  for (int i = 0; i < arr.size(); i++){
    sum += pow(arr[i],2);
  }
  return sum;
}

vector<double> trace_ray(vector<double> pos,double theta, double phi, double h){
  double v_x = cos(theta)*cos(phi);
  double v_y = sin(theta)*cos(phi);
  double v_z = sin(phi);
  double color[3] = {0,0,0};
  vector<double> vel = {v_x,v_y,v_z};
  double h2 = sqrsum(cross(pos,vel));
  Plane obj;
  while (norm(pos) <= 8) {
    double y[6] = {pos[0],pos[1],pos[2],v_x,v_y,v_z};
    double inc[6] = rk4(y,h2,h);
    double update[3] = {inc[0],inc[1],inc[2]};
    if (norm(add_arr(pos,update)) < 0.01){
      break;
    }
    if ((obj.p1[0] <= pos[1]+inc[1] <= obj.p2[0]) & (obj.p1[1] <= pos[2]+inc[2] <= obj.p2[1]) & (obj.r[0]-0.1 <= pos[0]+inc[0] <= obj.r[0]+0.1)){
      double *inds = obj.get_ind(pos[1]+inc[1],pos[2]+inc[2]);
      color[0] = inds[0];
      color[1] = inds[1];
      break;
    }
    pos = {pos[0]+inc[0],pos[1]+inc[1],pos[2]+inc[2]};
    v_x += inc[3];
    v_y += inc[4];
    v_z += inc[5];
  }
  return color;
}

double ***ray_cast(int w,int h){
  double cam[3] = {-10.0,0.0,0.0};
  double FOV_w = 40*3.14/180;
  double FOV_h = 30*3.14/180;
  double img[h][w][3];
  vector<double> t_ang = linspace(-FOV_h/2,FOV_h/2,h);
  vector<double> p_ang = linspace(-FOV_w/2,FOV_w/2,w);
  for (int i = 0; i < w; i++){
    for (int j = 0; j < h; j++){
      vector<double> vals = trace_ray(cam,t_ang[j],p_ang[i],0.1);
      img[j][i][0] = vals[0];
      img[j][i][1] = vals[1];
      img[j][i][2] = vals[2];
    }
  }
  return img;
}

int main(){
  int h = 9;
  int w = 16;
  double ***a = ray_cast(w,h);
  for (int i = 0; i < w; i++){
    for (int j = 0; j < h; j++){
      cout << a[j][i][0] << ", " << a[j][i][1] << ", " << a[j][i][2] << endl;
    }
  }
}

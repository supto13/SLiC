
#define ROW 64 // number of vertical channels in the sensor
#define COL 2000 // number of horizontal channels in the sensor

int totalMinute = 10; // total data duration
int nPoints = ROW * COL;
int nFrames = 600; // number of generated frames in each minute
double coeffThreshold = 1.0; // t_w
int neighSz = 30;
double simThreshold = 0.999; // t_c

std::string file_pref = "0000000000";

std::string loc = ""; //data frames location
std::string decompLoc = ""; // decompressed frames location




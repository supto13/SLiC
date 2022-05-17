#include <bits/stdc++.h>

#include "config.h"
#include "knncpp.h"
#include "wavelet.h"

#define eps 1e-15

typedef Eigen::MatrixXd Matrix;
typedef knncpp::Matrixi Matrixi;

using namespace Vector;

const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
const std::vector<double> Lo_D = {inv_sqrt2, inv_sqrt2};
const std::vector<double> Hi_D = {-inv_sqrt2, inv_sqrt2};
const std::vector<double> Lo_R = {inv_sqrt2, inv_sqrt2};
const std::vector<double> Hi_R = {inv_sqrt2, -inv_sqrt2};

// Instantiate a Wavelet object with Haar filters

const Wavelet<double> haar(Lo_D, Hi_D, Lo_R, Hi_R);

std::vector<double> data[3][ROW * COL], Intensity[ROW * COL];

std::vector<double> cxcs[3][ROW * COL];

std::vector<int> zeroIds, nzIds, keepIds;

int lvl = ceil(log2(nFrames));

std::vector<int> lvlLengths;

bool myIsnan(float v)
{
  std::uint32_t i;
  memcpy(&i, &v, 4);
  return ((i & 0x7f800000) == 0x7f800000) && (i & 0x7fffff);
}

void loadData(std::string file_name)
{

  /* allocate buffer to store the data */
  int32_t num = ROW * COL * 4;
  float *buf = (float *)malloc(num * sizeof(float));

  /* pointers */
  float *px = buf + 0;
  float *py = buf + 1;
  float *pz = buf + 2;
  float *pr = buf + 3;

  /* open file to read */
  FILE *stream = fopen(file_name.c_str(), "rb");
  // std::cout << file_name.c_str() << std::endl;

  num = fread(buf, sizeof(float), num, stream) / 4;

  for (int32_t i = 0; i < num; i++)
  {
    float val = std::move((*px));

    if (myIsnan(val))
    {
      val = 0;
    }

    data[0][i].push_back(val);
    val = std::move((*py));

    if (myIsnan(val))
    {
      val = 0;
    }

    data[1][i].push_back(val);
    val = std::move((*pz));

    if (myIsnan(val))
    {
      val = 0;
    }

    data[2][i].push_back(val);
    Intensity[i].push_back(std::move((*pr)));

    /* update the new point addresses */
    px += 4;
    py += 4;
    pz += 4;
    pr += 4;
  }
  /* free the additional space */
  free(buf);

  // outstream.close();
  fclose(stream);
}

void writeInCSRFormat(int nMinute)
{
  std::vector<std::string> dimName{"x", "y", "z"};
  std::ofstream IDS("ids.txt");
  std::string fileNames = "ids.txt";
  IDS << keepIds[0];
  for (int i = 1; i < (int)keepIds.size(); i++)
  {
    IDS << " " << keepIds[i];
  }
  IDS << std::endl;

  std::ofstream lvlout("levelLengths.txt");
  fileNames += " levelLengths.txt";
  lvlout << lvl;
  for (int i = lvl; i >= 0; i--)
  {
    lvlout << " " << lvlLengths[i];
  }
  lvlout << std::endl;

  for (int i = 0; i < 3; i++)
  {
    int nnz = 0;
    std::vector<int> nnzValues, colIds;
    std::vector<double> values;
    nnzValues.push_back(0);
    for (int j = 0; j < (int)keepIds.size(); j++)
    {
      int id = keepIds[j];
      for (int k = 0; k < (int)cxcs[i][id].size(); k++)
      {
        double val;
        val = cxcs[i][id][k];
        if (!(fabs(val) < eps))
        {
          nnz++;
          values.push_back(val);
          colIds.push_back(k);
        }
      }
      nnzValues.push_back(nnz);
    }
    std::string fileName = dimName[i] + "A.txt";
    std::ofstream A(fileName);
    fileNames += " " + fileName;
    A << std::setprecision(15) << values[0];
    for (int j = 1; j < (int)values.size(); j++)
      A << " " << std::setprecision(15) << values[j];
    A << std::endl;
    fileName = dimName[i] + "JA.txt";
    fileNames += " " + fileName;
    std::ofstream JA(fileName);
    JA << colIds[0];
    for (int j = 1; j < (int)colIds.size(); j++)
      JA << " " << colIds[j];
    JA << std::endl;
    fileName = dimName[i] + "IA.txt";
    fileNames += " " + fileName;
    std::ofstream IA(fileName);
    IA << nnzValues[0];
    for (int j = 1; j < (int)nnzValues.size(); j++)
      IA << " " << nnzValues[j];
    IA << std::endl;
  }
  std::string cmd = "tar -czf frames" + std::to_string(nMinute) + ".tar.gz" + " " + fileNames;
  if (system(cmd.c_str()) == -1)
  {
    std::cout << "[ERROR]: 'tar' command compression failed." << std::endl;
    exit(-1);
  }

  std::string rm_files = "rm " + fileNames;
  if (system(rm_files.c_str()) == -1)
  {
    std::cout << "[ERROR]: 'rm' command remove file failed." << std::endl;
    exit(-1);
  }
}

inline double ddencmp(std::vector<double> &x)
{
  double thr = 1;
  Decomposition1D<double> dec = haar.Wavedec(x, 1);
  std::vector<double> c = dec.GetDetcoef(0);
  std::vector<double> md;
  int sz = (int)c.size();
  for (int i = 0; i < sz; i++)
  {
    md.push_back(fabs(c[i]));
  }
  std::sort(md.begin(), md.end());
  double normaliz;
  if (sz & 1)
  {
    normaliz = md[sz >> 1];
  }
  else
  {
    normaliz = (md[sz >> 1] + md[(sz >> 1) - 1]) / 2;
  }
  if (fabs(normaliz) < eps)
  {
    double mx = *std::max_element(md.begin(), md.end());
    normaliz = 0.05 * mx;
  }
  thr = thr * normaliz;
  return thr;
}

void wdencmp(std::vector<double> &x, double thr, bool &flag, std::vector<double> &cxc)
{
  Decomposition1D<double> dec = haar.Wavedec(x, lvl);
  std::vector<double> app = dec.GetAppcoef();
  if (flag)
  {
    lvlLengths.at(lvl) = app.size();
  }
  int sz = (int)app.size();
  for (int i = 0; i < sz; i++)
  {
    if (fabs(app[i]) >= coeffThreshold)
    {
      cxc.push_back(app[i]);
    }
    else
    {
      cxc.push_back(0);
    }
  }
  for (int i = lvl - 1; i >= 0; i--)
  {
    std::vector<double> coef = dec.GetDetcoef(i);
    if (flag)
    {
      lvlLengths.at(i) = coef.size();
    }
    sz = (int)coef.size();
    for (int j = 0; j < sz; j++)
    {
      if (fabs(coef[j]) > thr && fabs(coef[j]) >= coeffThreshold)
      {
        cxc.push_back(coef[j]);
      }
      else
      {
        cxc.push_back(0);
      }
    }
  }
}

void zNorm(std::vector<double> x, std::vector<double> &y)
{
  double ex = 0, ex2 = 0;
  int n = x.size();
  for (int i = 0; i < n; i++)
  {
    ex += x[i];
    ex2 += x[i] * x[i];
  }
  double mean = ex / n;
  double std = ex2 / n;
  std = sqrt(std - mean * mean);
  for (int i = 0; i < n; i++)
    y.push_back((x[i] - mean) / std);
}

void compression(int nMinute)
{
  bool flag = true;
  for (int i = 0; i < nPoints; i++)
  {
    int cnt = 0;
    for (int j = 0; j < nFrames; j++)
    {
      if (fabs(data[0][i][j]) < eps)
        cnt++;
    }
    if (cnt == nFrames)
    {
      zeroIds.push_back(i);
      continue;
    }
    for (int j = 0; j < 3; j++)
    {
      double thr = ddencmp(data[j][i]);
      // std::cout << thr << std::endl;
      wdencmp(data[j][i], thr, flag, cxcs[j][i]);
      flag = false;
    }
    cnt = 0;
    int sz = (int)cxcs[0][i].size();
    for (int j = 0; j < 3; j++)
    {
      for (int k = 0; k < sz; k++)
      {
        if (fabs(cxcs[j][i][k]) < eps)
          cnt++;
      }
    }
    if (cnt == 3 * sz)
    {
      zeroIds.push_back(i);
    }
    else
    {
      nzIds.push_back(i);
    }
  }

  int coefSz = 50;

  for (int i = 0; i < ROW; i++)
  {
    int st = i * COL;
    int ed = (i + 1) * COL - 1;
    std::vector<int>::iterator low, up;
    low = std::lower_bound(nzIds.begin(), nzIds.end(), st);
    up = std::upper_bound(nzIds.begin(), nzIds.end(), ed);
    int sz = up - nzIds.begin();
    sz -= (low - nzIds.begin());
    if (sz <= 0)
      continue;

    Matrix dataPoints(coefSz * 3, sz);
    int k = 0;
    std::vector<int> ids;
    for (std::vector<int>::iterator it = low; it != up; it++)
    {
      int id = *it;
      ids.push_back(id);
      std::vector<double> x, y;
      for (int j = 0; j < coefSz; j++)
        x.push_back(cxcs[0][id][j]);
      for (int j = 0; j < coefSz; j++)
        x.push_back(cxcs[1][id][j]);
      for (int j = 0; j < coefSz; j++)
        x.push_back(cxcs[2][id][j]);
      zNorm(x, y);
      for (int j = 0; j < 3 * coefSz; j++)
        dataPoints(j, k) = y[j];
      k++;
    }
    knncpp::KDTreeMinkowskiX<double, knncpp::EuclideanDistance<double>> kdtree(dataPoints);
    kdtree.setBucketSize(50);
    kdtree.setThreads(8);
    kdtree.setTakeRoot(true);
    kdtree.build();

    Matrixi indices;
    Matrix distances;

    // Their value is set to -1 if no further neighbor was found.
    kdtree.query(dataPoints, neighSz + 1, indices, distances);

    double dThreshold = sqrt((1.0 - simThreshold) * 150);

    bool isTaken[sz];

    for (int j = 0; j < sz; j++)
      isTaken[j] = true;

    for (int j = 0; j < sz; j++)
    {
      if (isTaken[j])
      {
        keepIds.push_back(ids[j]);
        for (int k = 1; k < neighSz + 1; k++)
        {
          int id = indices(k, j);
          if (id == -1)
            continue;
          if (distances(k, j) <= dThreshold)
          {
            isTaken[id] = false;
          }
        }
      }
    }
  }
  writeInCSRFormat(nMinute);
}

void clearall()
{
  for (int i = 0; i < ROW * COL; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      data[j][i].clear();
      cxcs[j][i].clear();
    }
    Intensity[i].clear();
  }
  zeroIds.clear();
  nzIds.clear();
  keepIds.clear();
  lvlLengths.clear();
}
int main()
{

  for (int nMinute = 1; nMinute <= totalMinute; nMinute++)
  {
    clearall();
    for (int i = 0; i < lvl + 1; i++)
      lvlLengths.push_back(0);
    for (int i = 1; i <= 600; i++)
    {
      std::string suf = std::to_string((nMinute - 1) * 600 + i);
      int sz = 10 - suf.size();
      suf += ".bin";
      std::string file_name = loc;
      for (int j = 0; j < sz; j++)
        file_name += file_pref[j];
      file_name += suf;
      loadData(file_name);
    }

    unsigned long long start_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    compression(nMinute);

    unsigned long long end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    printf("%lf\n", (end_time - start_time) / 1000.0);
  }

  return 0;
}

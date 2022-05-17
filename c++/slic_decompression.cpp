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

std::vector<double> cxcs[3][ROW * COL];

std::vector<int> keepIds;

std::vector<double> dxcs[3][ROW * COL], frame[ROW * COL];

int lvl = ceil(log2(nFrames));

std::vector<int> lvlLengths(lvl + 1, 0);

template <typename Real>
int nearestNeighbourIndex(std::vector<Real> &x, Real &value)
{
  Real dist = std::numeric_limits<Real>::max();
  Real newDist = dist;
  size_t idx = 0;

  for (size_t i = 0; i < x.size(); ++i)
  {
    newDist = std::abs(value - x[i]);
    if (newDist <= dist)
    {
      dist = newDist;
      idx = i;
    }
  }

  return idx;
}

template <typename Real>
std::vector<Real> interp1(std::vector<Real> &x, std::vector<Real> &y, std::vector<Real> &x_new)
{
  std::vector<Real> y_new;
  Real dx, dy, m, b;
  size_t x_max_idx = x.size() - 1;
  size_t x_new_size = x_new.size();

  y_new.reserve(x_new_size);

  for (size_t i = 0; i < x_new_size; ++i)
  {
    size_t idx = nearestNeighbourIndex(x, x_new[i]);

    if (x[idx] > x_new[i])
    {
      dx = idx > 0 ? (x[idx] - x[idx - 1]) : (x[idx + 1] - x[idx]);
      dy = idx > 0 ? (y[idx] - y[idx - 1]) : (y[idx + 1] - y[idx]);
    }
    else
    {
      dx = idx < x_max_idx ? (x[idx + 1] - x[idx]) : (x[idx] - x[idx - 1]);
      dy = idx < x_max_idx ? (y[idx + 1] - y[idx]) : (y[idx] - y[idx - 1]);
    }

    m = dy / dx;
    b = y[idx] - x[idx] * m;

    y_new.push_back(x_new[i] * m + b);
  }

  return y_new;
}

bool myIsnan(float v)
{
  std::uint32_t i;
  memcpy(&i, &v, 4);
  return ((i & 0x7f800000) == 0x7f800000) && (i & 0x7fffff);
}

void convertCSRToDense(std::string compressedFileName)
{
  std::string cmd = "tar -xzf " + compressedFileName;
  if (system(cmd.c_str()) == -1)
  {
    std::cout << "[ERROR]: 'tar' command decompression failed." << std::endl;
    exit(-1);
  }
  std::vector<std::string> dimName{"x", "y", "z"};
  std::ifstream IDS("ids.txt");
  std::string fileNames = "ids.txt";
  keepIds.clear();
  int id;
  while (IDS >> id)
  {
    keepIds.push_back(id);
  }
  std::ifstream lvlin("levelLengths.txt");
  fileNames += " levelLengths.txt";
  lvlLengths.clear();
  for (int i = 0; i < lvl + 1; i++)
    lvlLengths.push_back(0);
  int len;
  int nCoeff = 0;
  lvlin >> len;
  int i = lvl;
  while (lvlin >> len)
  {
    nCoeff += len;
    lvlLengths.at(i) = len;
    i--;
  }
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < nPoints; j++)
      cxcs[i][j].clear();
  }
  for (int i = 0; i < 3; i++)
  {
    std::vector<double> values;
    std::vector<int> colIds, nnzValues;
    double val;
    int nnz, c;
    std::string fileName = dimName[i] + "A.txt";
    fileNames += " " + fileName;
    std::ifstream A(fileName);
    while (A >> val)
    {
      values.push_back(val);
    }
    fileName = dimName[i] + "IA.txt";
    fileNames += " " + fileName;
    std::ifstream IA(fileName);
    while (IA >> nnz)
    {
      nnzValues.push_back(nnz);
    }
    fileName = dimName[i] + "JA.txt";
    fileNames += " " + fileName;
    std::ifstream JA(fileName);
    while (JA >> c)
    {
      colIds.push_back(c);
    }
    for (int j = 0; j < (int)keepIds.size(); j++)
    {
      int id = keepIds[j];

      for (int k = 0; k < nCoeff; k++)
      {
        cxcs[i][id].push_back(0);
      }
      int rowStart = nnzValues[j];
      int rowEnd = nnzValues[j + 1];
      for (int k = rowStart; k < rowEnd; k++)
      {
        cxcs[i][id].at(colIds[k]) = values[k];
      }
    }
  }
  std::string rm_files = "rm " + fileNames;
  if (system(rm_files.c_str()) == -1)
  {
    std::cout << "[ERROR]: 'rm' command remove file failed." << std::endl;
    exit(-1);
  }
}

void writeFrame(std::string fileName)
{
  /* pointers */
  float x, y, z, r;
  r = 0;

  /* open file to read */
  std::ofstream out_stream(fileName, std::ofstream::binary);
  // std::cout << "Write " << filename << std::endl;

  for (int i = 0; i < nPoints; i++)
  {
    x = frame[i][0];
    y = frame[i][1];
    z = frame[i][2];
    out_stream.write(reinterpret_cast<const char *>(&x), sizeof(x));
    out_stream.write(reinterpret_cast<const char *>(&y), sizeof(y));
    out_stream.write(reinterpret_cast<const char *>(&z), sizeof(z));
    out_stream.write(reinterpret_cast<const char *>(&r), sizeof(r));
  }
  out_stream.close();
}
void decompression(int nMinute)
{
  std::string compressedFileName = "frames" + std::to_string(nMinute) + ".tar.gz";

  convertCSRToDense(compressedFileName);

  for (int i = 0; i < (int)keepIds.size(); i++)
  {
    int id = keepIds[i];
    for (int dim = 0; dim < 3; dim++)
    {
      Decomposition1D<double> dec(lvl);
      int cm = 0;
      for (int j = lvl; j >= 0; j--)
      {
        std::vector<double> coef;
        for (int k = 0; k < lvlLengths[j]; k++)
        {
          coef.push_back(cxcs[dim][id][cm + k]);
        }
        if (j == lvl)
        {
          dec.SetAppcoef(coef);
        }
        else
        {
          dec.SetDetcoef(coef, j);
        }
        cm += lvlLengths[j];
      }
      std::vector<double> rec = haar.Waverec(dec, nFrames);
      for (int j = 0; j < nFrames; j++)
      {
        dxcs[dim][id].push_back(rec[j]);
      }
    }
  }

  /*

  for (int i = 1; i <= nFrames; i++)
  {
    std::string suf = std::to_string((nMinute - 1) * nFrames + i);
    int sz = 10 - suf.size();
    suf += ".bin";
    std::string fileName = decompLoc;
    for (int j = 0; j < sz; j++)
      fileName += file_pref[j];
    fileName += suf;
    for (int j = 0; j < nPoints; j++)
    {
      frame[j].clear();
      for (int k = 0; k < 3; k++)
      {
        if (dxcs[k][j].size() > 0)
        {
          frame[j].push_back(dxcs[k][j][i - 1]);
        }
        else
        {
          frame[j].push_back(0);
        }
      }
    }
    writeFrame(fileName);
  }
  */
  

  int h = ROW/2;
  int h1 = h/2;

  for (int i = 0; i < nFrames; i++)
  {
    int z = 0;
    for (int j = 0; j < ROW; j++)
    {
      for (int k = 0; k < COL; k++)
      {
        frame[z].clear();
        z++;
      }
    }
    z = 0;
    for (int j = 0; j < ROW; j++)
    {
      std::vector<double> pointsX(COL, NAN);
      std::vector<double> pointsY(COL, NAN);
      std::vector<double> pointsZ(COL, NAN);
      for (int k = 0; k < COL; k++)
      {
        if ((int)dxcs[0][z].size() > 0)
        {
          pointsX.at(k) = dxcs[0][z][i];
          pointsY.at(k) = dxcs[1][z][i];
          pointsZ.at(k) = dxcs[2][z][i];
        }
        z++;
      }
      int st = 0;
      while (myIsnan(pointsX[st]) && st < COL)
      {
        st = st + 1;
        if (st < COL)
        {
          pointsX.at(st) = 0;
          pointsY.at(st) = 0;
          pointsZ.at(st) = 0;
        }
      }
      int nxt = st + 1;
      while (nxt < COL)
      {
        if (!myIsnan(pointsX[nxt]))
        {
          double dis = (pointsX[st] - pointsX[nxt]) * (pointsX[st] - pointsX[nxt]) + (pointsY[st] - pointsY[nxt]) * (pointsY[st] - pointsY[nxt]);
          dis += (pointsZ[st] - pointsZ[nxt]) * (pointsZ[st] - pointsZ[nxt]);
          dis = sqrt(dis);
          if ((j < h && dis > 0.8) || (j < (h+h1) && dis > 3) || (j >= (h+h1) && dis > 5))
          // if ((j < 8 && dis > 0.8) || (j < 12 && dis > 3) || (j >= 12 && dis > 5))
          {
            for (int k = st + 1; k < nxt - 1; k++)
            {
              pointsX.at(k) = 0;
              pointsY.at(k) = 0;
              pointsZ.at(k) = 0;
            }
          }
          st = nxt;
        }
        nxt = nxt + 1;
      }
      for (int k = st + 1; k < nxt - 1; k++)
      {
        pointsX.at(k) = 0;
        pointsY.at(k) = 0;
        pointsZ.at(k) = 0;
      }

      std::vector<double> x, valx, valy, valz, newX;
      for (int k = 0; k < COL; k++)
      {
        if (!myIsnan(pointsX[k]))
        {
          x.push_back(k + 1);
          valx.push_back(pointsX[k]);
          valy.push_back(pointsY[k]);
          valz.push_back(pointsZ[k]);
        }
        else
          newX.push_back(k + 1);
      }
      std::vector<double> interpxVal = interp1(x, valx, newX);
      std::vector<double> interpyVal = interp1(x, valy, newX);
      std::vector<double> interpzVal = interp1(x, valz, newX);

      for (int k = 0; k < (int)newX.size(); k++)
      {
        int p = newX[k] - 1;
        pointsX.at(p) = interpxVal[k];
        pointsY.at(p) = interpyVal[k];
        pointsZ.at(p) = interpzVal[k];
      }
      int kk = j * COL;
      for (int k = 0; k < COL; k++)
      {
        frame[kk].push_back(pointsX[k]);
        frame[kk].push_back(pointsY[k]);
        frame[kk++].push_back(pointsZ[k]);
      }
    }
    
    std::string suf = std::to_string((nMinute - 1) * nFrames + i + 1);
    int sz = 10 - suf.size();
    suf += ".bin";
    std::string fileName = decompLoc;
    for (int j = 0; j < sz; j++)
      fileName += file_pref[j];
    fileName += suf;
    writeFrame(fileName);
  }
  
}
void clearall()
{
  for (int i = 0; i < ROW * COL; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      dxcs[j][i].clear();
      cxcs[j][i].clear();
    }
  }
  keepIds.clear();
  lvlLengths.clear();
}
int main()
{

  for (int nMinute = 1; nMinute <= totalMinute; nMinute++)
  {
    clearall();
    unsigned long long dcomStart = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    decompression(nMinute);
    unsigned long long dcomEnd = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    printf("%lf\n", (dcomEnd - dcomStart) / 1000.0);
  }

  return 0;
}

#include "timing.h"

double getTimeStamp()
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

double getTimeResolution()
{
  struct timespec ts;
  clock_getres(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9;
}

double getTimeStamp_()
{
  return getTimeStamp();
}

double gettimestamp_()
{
  return getTimeStamp();
}


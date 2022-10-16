#pragma once
#ifndef ING_TIMER_H
#define ING_TIMER_H

#include <ing/common.h>

ING_NAMESPACE_BEGIN

class Timer {
public:
    Timer();
    float TotalTime() const; // in seconds
    float DeltaTime() const; // in seconds

    void Reset();
    void Start();
    void Stop();
    void Tick();
private:
    double mSecondsPerCount;
    double mDeltaTime;

    __int64 mBaseTime;
    __int64 mPausedTime;
    __int64 mStopTime;
    __int64 mPrevTime;
    __int64 mCurrTime;

    bool mStopped;
};

ING_NAMESPACE_END

#endif // ING_TIMER_H
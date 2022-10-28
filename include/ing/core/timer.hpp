#pragma once
#ifndef ING_CORE_TIMER_H_
#define ING_CORE_TIMER_H_

#include <ing/common.h>
#include <chrono>

ING_NAMESPACE_BEGIN

typedef std::chrono::time_point<std::chrono::high_resolution_clock> INGTimePoint;

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
    float mDeltaTime;

    INGTimePoint mBaseTime;
    float mPausedTime;
    INGTimePoint mStopTime;
    INGTimePoint mPrevTime;
    INGTimePoint mCurrTime;

    bool mStopped;
};

ING_NAMESPACE_END

#endif // ING_CORE_TIMER_H
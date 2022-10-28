#include <ing/core/timer.hpp>

ING_NAMESPACE_BEGIN

Timer::Timer():
    mDeltaTime(-1.0),
    mStopped(false)
{
    INGTimePoint now = std::chrono::high_resolution_clock::now();
    mBaseTime = now;
    mPausedTime = 0.0;
    mPrevTime = now;
    mCurrTime = now;
}

float Timer::TotalTime() const 
{
    if (mStopped)
    {
        
        return (float)std::chrono::duration<float, std::chrono::seconds::period>(mStopTime  - mBaseTime).count() - mPausedTime;
    }
    else
    {
        
        return (float)std::chrono::duration<float, std::chrono::seconds::period>(mCurrTime - mBaseTime).count() - mPausedTime;
    }
}

float Timer::DeltaTime() const
{
    return (float)mDeltaTime;
}

void Timer::Reset()
{
    INGTimePoint now = std::chrono::high_resolution_clock::now();
    mBaseTime = now;
    mPrevTime = now; 
    mStopped = false; 
}

void Timer::Start()
{
    INGTimePoint now = std::chrono::high_resolution_clock::now();
    if (mStopped)
    {
        mPausedTime = mPausedTime + (float)std::chrono::duration<float, std::chrono::seconds::period>(now - mStopTime).count();
        mPrevTime = now;
        // mStopTime = 0;
        mStopped = false;
    }
}

void Timer::Stop()
{
    if (!mStopped) {
       INGTimePoint now = std::chrono::high_resolution_clock::now();

       mStopTime = now;
       mStopped = true; 
    }
}

void Timer::Tick()
{
    if (mStopped)
    {
        mDeltaTime = 0.0;
        return;
    }

    INGTimePoint now = std::chrono::high_resolution_clock::now();
    mCurrTime = now;

    mDeltaTime = (float)std::chrono::duration<float, std::chrono::seconds::period>(mCurrTime - mPrevTime).count();
    mPrevTime = mCurrTime;

    if (mDeltaTime < 0.0)
    {
        mDeltaTime = 0.0;
    }
}

ING_NAMESPACE_END
